import pynini
import pandas as pd
import json

### ENCODING PARAMETERS
ENCODING       = 'utf-8'
NULL_TOKEN     = '<e>'
SPACE_TOKEN    = '<s>'
COMPLETE_TOKEN = '<t>'

### TOKEN + GRAMMAR FILES
LETTERS_FILE_PATH            = "src/FST/tokens/letters.json"
TEX_LETTERS_FILE_PATH        = "src/FST/tokens/tex_letters.json"
TEX_GRAMMAR_FILE_PATH        = "src/FST/grammar/tex_grammar.json"

### RULE FILES 
NORMALIZER_RULE_FILE         = "src/FST/rules/Normalizer/normalizer.txt"
NUMBERS_NORMALIZER_RULE_FILE = "src/FST/rules/Normalizer/numbers.txt"
LEMMATIZER_RULE_FILE         = "src/FST/rules/Lemmatizer/lemmatizer.txt"
MATH_LEX_RULE_FILE           = "src/FST/rules/LexicalRules/math_rules.txt"
MATH_LEX_KSEQ_RULE_FILE      = "src/FST/rules/LexicalRules/math_rules_keyseq.txt"
TEX_NORMALIZER_RULE_FILE     = "src/FST/rules/TexNormalizer/tex_normalizer.txt"
MATH_GRA_RULE_RILE           = "src/FST/rules/GrammarRules/grammar_rules.txt"

with open(LETTERS_FILE_PATH, 'r', encoding=ENCODING) as reader :
    DICT_LETTERS = json.load(reader)

with open(TEX_LETTERS_FILE_PATH, 'r', encoding=ENCODING) as reader :
    DICT_TEX_LETTERS = json.load(reader)

with open(TEX_GRAMMAR_FILE_PATH, 'r', encoding=ENCODING) as reader :
    TEX_GRAMMAR = json.load(reader)

# *****************************************************************
#
#                       BUILD ACCEPTORS
#
# *****************************************************************

#### BUILD Sigma_in and Sigma_tex
Sig_seq = pynini.SymbolTable()
Sig_seq.add_symbol(NULL_TOKEN, key=0)
Sig_seq.add_symbol(SPACE_TOKEN, key=1)
Sig_seq.add_symbol(COMPLETE_TOKEN, key=2)

Sig_tex = pynini.SymbolTable()
Sig_tex.add_symbol(NULL_TOKEN, key=0)
Sig_tex.add_symbol(SPACE_TOKEN, key=1)
Sig_tex.add_symbol(COMPLETE_TOKEN, key=2)

#### BUILD Sigma_in and Sigma_tex acceptor
Sig_seq_acc = pynini.Fst()
Sig_tex_acc = pynini.Fst()

for letter_key in DICT_LETTERS.keys() : 
    for letter in DICT_LETTERS[letter_key] : 
        Sig_seq.add_symbol(letter)
        Sig_seq_acc = pynini.union(Sig_seq_acc, pynini.accep(letter, token_type=Sig_seq))

for tex_key in DICT_TEX_LETTERS.keys() :
    for letter in DICT_TEX_LETTERS[tex_key] :
        Sig_tex.add_symbol(letter)
        Sig_tex_acc = pynini.union(Sig_tex_acc, pynini.accep(letter, token_type=Sig_tex))

#### BUILD sub-Sigma_tex acceptor
def build_tex_acceptor(tex_symbol_primitive_types):
    tex_sub_acc = pynini.Fst()
    for s in tex_symbol_primitive_types : 
        for name in TEX_GRAMMAR[s].keys() :
            acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
            for sub_wrd in name.split():
                acc = pynini.concat(acc, pynini.accep(sub_wrd, token_type=Sig_tex))
            tex_sub_acc = pynini.union(tex_sub_acc, acc)
    return pynini.difference(tex_sub_acc, pynini.accep(NULL_TOKEN, token_type=Sig_tex))

GRAMMAR_ACC = {

    "all":Sig_seq_acc,
    "var":build_tex_acceptor(['var_name']).optimize(),
    "num":build_tex_acceptor(['digit']).optimize(),
    "inf":build_tex_acceptor(['infty']).optimize(),
    "set":build_tex_acceptor(['sets']).optimize(),
    "rel":build_tex_acceptor(['rel']).optimize(),
    "qtf":build_tex_acceptor(['quantif']).optimize(),
    "uop":build_tex_acceptor(['un_op']).optimize(),
    "bop":build_tex_acceptor(['bin_op']).optimize(),
    "sub":build_tex_acceptor(['subscript']).optimize(),
    "sup":build_tex_acceptor(['supscript']).optimize(),
    "fun":build_tex_acceptor(['func_names', 'var_name']).optimize(),
    "fun_cseq":build_tex_acceptor(['func_cseq']).optimize(),
    "symb":build_tex_acceptor(['symb']).optimize(),
}




# *****************************************************************
#
#                  NORM + LEMM + ANALYSE LEXICALE
#
# *****************************************************************

class RulesTransducer():

    # **********************************
    # *          Construct             *
    # **********************************

    def __init__(self, 
        output_word:bool=True, wrd_token_sep:str=" ", 
        complete_mod:bool=True, trash_mod:bool=True, weight_mod:bool=True):
        """
        -- initialize rules transducer structure D(RD)*

        >> In : 
            * output_word:bool = True                   - if output has to be considered as list of words or list of characters  
            * wrd_token_sep:str = " "                   - separator symbol of output words list 
            * complete_mod:bool = True                  - if inputs have to be recognized as complete words (if True : it will add separator before and after words)
            * trash_mod:bool = True                     - if words recognized by D are removed or not
            * weight_mod:bool = True                    - if rules are weighted (counting transitions in R)

        << 
        """

        # transducers
        self.__fst = pynini.Fst()
        self.__R = pynini.Fst()
        self.__Rw = pynini.Fst()
        self.__update = True

        # properties
        self.__complete_mod = complete_mod
        self.__trash_mod = trash_mod
        self.__weight_mod = weight_mod
        self.__output_word = output_word
        self.__wrd_token_sep = wrd_token_sep

        # tokens 
        self.__input_tokens = pynini.SymbolTable()
        self.__input_tokens.add_symbol(NULL_TOKEN, key = 0)
        self.__input_tokens.add_symbol(SPACE_TOKEN, key = 1)
        self.__input_tokens.add_symbol(COMPLETE_TOKEN, key = 2)

        self.__output_tokens = pynini.SymbolTable()
        self.__output_tokens.add_symbol(NULL_TOKEN, key = 0)
        self.__output_tokens.add_symbol(SPACE_TOKEN, key = 1)
        self.__output_tokens.add_symbol(COMPLETE_TOKEN, key = 2)

        L_star = pynini.accep(SPACE_TOKEN, token_type=self.__input_tokens)
        L_star = pynini.union(L_star, pynini.accep(COMPLETE_TOKEN, token_type=self.__input_tokens))
        for key in DICT_LETTERS.keys():
            for char in DICT_LETTERS[key]:
                self.__input_tokens.add_symbol(char)
                if not(self.__output_word) or not(self.__trash_mod): self.__output_tokens.add_symbol(char)
                L_star = pynini.union(L_star, pynini.accep(char, token_type=self.__input_tokens))
        self.L_star = pynini.determinize(L_star).closure().optimize()



    # **********************************
    # *          Properties            *
    # **********************************

    def get_fst(self):
        if not(self.__update) :self.__build_fst()
        return self.__fst
    
    def R(self):
        return self.__R.set_input_symbols(self.input_tokens()).set_output_symbols(self.output_tokens())
    
    def Rw(self):
        return self.__Rw.set_input_symbols(self.input_tokens()).set_output_symbols(self.output_tokens())
    
    def input_tokens(self, dic=False):
        if dic : return dict(self.__input_tokens)
        else : return self.__input_tokens
    
    def output_tokens(self, dic=False):
        if dic : return dict(self.__output_tokens)
        else : return self.__output_tokens

    def accep(self, input_string:str):
        symbols = self.get_input_symbol_list(input_string)
        acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokens)
        for s in symbols : 
            acc = pynini.concat(acc, pynini.accep(s, token_type = self.__input_tokens))
        return acc.optimize()

    def get_input_symbol_list(self, string:str):
        symbols = []
        if self.__complete_mod :
            symbols.append(COMPLETE_TOKEN)
            for i in string :
                if  i == " " : 
                    symbols.append(COMPLETE_TOKEN)
                    symbols.append(SPACE_TOKEN)
                    symbols.append(COMPLETE_TOKEN)
                else :
                    symbols.append(i)
            symbols.append(COMPLETE_TOKEN)
        else :
            for i in string :
                if  i == " " : 
                    symbols.append(SPACE_TOKEN)
                else :
                    symbols.append(i)
        return symbols

    def get_output_symbol_list(self, string:str):
        if self.__output_word : 
            return string.split(self.__wrd_token_sep)
        else : 
            output_symbols = []
            for i in string :
                if i == " " : output_symbols.append(SPACE_TOKEN)
                else : output_symbols.append(i)

    def predict(self, input_sentence, clean_space=True):
        acceptor = self.accep(input_sentence)
        res = pynini.shortestpath(acceptor @ self.get_fst()).paths(output_token_type = self.__output_tokens).ostring()
        if not(self.__output_word): res = res.replace(' ','') 
        if self.__complete_mod : res = res.replace(COMPLETE_TOKEN,'')
        res = res.replace(SPACE_TOKEN, " ")
        if clean_space : res = " ".join(res.split())
        return res


    # **********************************
    # *            Setters             *
    # **********************************

    def add_rule(self, input_string, output_string):
        
        self.__update = False
        
        input_symbols = self.get_input_symbol_list(input_string)

        if self.__output_word : 
            output_symbols = output_string.split(self.__wrd_token_sep)
            for wrd in output_symbols :
                if not(wrd in dict(self.__output_tokens).values()) : self.__output_tokens.add_symbol(wrd)
        else : 
            output_symbols = []
            for i in output_string :
                if i == " ":output_symbols.append(SPACE_TOKEN)
                else :output_symbols.append(i)
            
        w = -1 if self.__weight_mod else 0 
        size = len(input_symbols) - 1

        acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokens, weight=0)
        acc_w = pynini.accep(NULL_TOKEN, token_type=self.__input_tokens, weight=w)
        out_acc = pynini.accep(NULL_TOKEN, token_type=self.__output_tokens)
        
        for i, s in enumerate(input_symbols):
            self.__input_tokens.add_symbol(s)
            acc = pynini.concat(acc, pynini.accep(s, token_type=self.__input_tokens))
            acc_w = pynini.concat(acc_w, pynini.accep(s, token_type=self.__input_tokens, weight=(w)*(i!=size)))

        for i, s in enumerate(output_symbols):
            out_acc = pynini.concat(out_acc, pynini.accep(s, token_type=self.__output_tokens))
        
        self.__R = pynini.union(self.__R, pynini.cross(acc.optimize(), out_acc.optimize()))
        self.__Rw = pynini.union(self.__Rw, pynini.cross(acc_w.optimize(), out_acc.optimize()))

    def add_rules(self, csv_file_path, sep=';', header=True):
        self.__update = False
        df = pd.read_csv(csv_file_path, sep=sep)
        inputs = list(df['INPUT'])
        outputs = list(df['OUTPUT'])
        for i, input in enumerate(inputs):
            output = outputs[i]
            self.add_rule(input, output)

    # **********************************
    # *          SUB METHODS           *
    # **********************************

    def __build_fst(self) :
        D = pynini.determinize(pynini.difference(self.L_star, pynini.concat(self.L_star, pynini.concat(self.__R.copy().project('input'), self.L_star)))).optimize()
        if self.__trash_mod : D = pynini.cross(D, pynini.accep(NULL_TOKEN, token_type=self.__output_tokens))
        self.__fst = pynini.concat(D, pynini.concat(self.__Rw, D).closure()).optimize()
        self.__update = True

class Normalizer():

    def __init__(self,
                 normalize_rule_file = NORMALIZER_RULE_FILE, comp_norm_rule_file = NUMBERS_NORMALIZER_RULE_FILE, 
                 sep_rule=';', header=True):
        
        self.__normalizer      = RulesTransducer(output_word=False, complete_mod=False, trash_mod=False)
        self.__comp_normalizer = RulesTransducer(output_word=False, complete_mod=True, trash_mod=False)
        self.__use_comp_normalizer = (comp_norm_rule_file != None)

        with open(normalize_rule_file, 'r', encoding=ENCODING) as norm_file:
            for i, line in enumerate(norm_file): 
                if not(header and i == 0) : 
                    s = line.split(';')
                    _in, _out = s[0], s[1]
                    self.__normalizer.add_rule(_in, _out)
        
        if self.__use_comp_normalizer :
            with open(comp_norm_rule_file, 'r', encoding=ENCODING) as comp_file:
                for i, line in enumerate(comp_file):
                    if not(header and i == 0) : 
                        s = line.split(';')
                        _in, _out = s[0], s[1]
                        self.__comp_normalizer.add_rule(_in, _out)

    def predict(self, input_string:str):
        input_string = self.__normalizer.predict(input_string)
        if self.__use_comp_normalizer :
            input_string = self.__comp_normalizer.predict(input_string)
        return input_string

class Lemmatizer():
    
    def __init__(self, 
                 lemmatize_rule_file = LEMMATIZER_RULE_FILE, sep_rule=';', header=True):
        
        self.__lemmatizer = RulesTransducer(output_word=False, complete_mod=True, trash_mod=False)

        with open(lemmatize_rule_file, 'r', encoding=ENCODING) as lemm_file : 
            for i, line in enumerate(lemm_file):
                if not(header and i == 0):
                    s = line.split(';')
                    _in, _out = s[0], s[1]
                    self.__lemmatizer.add_rule(_in, _out)

    def predict(self, input_string:str):
        input_string = self.__lemmatizer.predict(input_string)
        return input_string

class LexMathTransducer():
    
    def __init__(self, math_rule_file=MATH_LEX_RULE_FILE, 
                 normalize_rule_file=True, lemmatize_rule_file=True, 
                 sep_rule=';', header=True):
        
        self.__normalizer = Normalizer()
        self.__lemmatizer = Lemmatizer()
        self.__lexmathfst = RulesTransducer(output_word=True, complete_mod=True, trash_mod=True)

        self.__normalize_rules = normalize_rule_file
        self.__lemmatize_rules = lemmatize_rule_file

        with open(math_rule_file, 'r', encoding=ENCODING) as math_file:
            for i, line in enumerate(math_file):
                if not(header and i == 0) :
                    s = line.split(';')
                    _in, _out = s[0], s[1]
                    if normalize_rule_file : _in = self.__normalizer.predict(_in)
                    if lemmatize_rule_file : _in = self.__lemmatizer.predict(_in)
                    self.__lexmathfst.add_rule(_in, _out)

    def predict(self, input_string:str, trace=False):
        
        if self.__normalize_rules : input_string = self.__normalizer.predict(input_string)
        if trace : print(f"  >> norm = [{input_string}]")
        
        if self.__lemmatize_rules : input_string = self.__lemmatizer.predict(input_string)
        if trace : print(f"  >> lemm = [{input_string}]")
        
        input_string = self.__lexmathfst.predict(input_string)
        if trace : print(f"  >> math = [{input_string}]")
        
        return input_string

class TexNormalizer():

    def __init__(self, tex_normalizer_file=TEX_NORMALIZER_RULE_FILE, sep_rule=';', header=True):
        
        self.__normalizer = RulesTransducer(output_word=False, complete_mod=False, trash_mod=False)
        
        with open(tex_normalizer_file, 'r', encoding=ENCODING) as rule_file:
            for i, line in enumerate(rule_file):
                if not(header and i == 0) :
                    s = line.split(';')
                    _in, _out = s[0], s[1]
                    self.__normalizer.add_rule(_in, _out)

    def predict(self, input_string:str):
        return self.__normalizer.predict(input_string)


# *****************************************************************
#
#                           GRAMMAIRE
#
# *****************************************************************

class GrammarRuleAcceptor():

    def __init__(self, input_grammar, output_grammar):

        input_symbols = input_grammar.split()
        output_symbols = output_grammar.split()

        acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
        in_acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
        out_acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
        out_idx = 0

        for  in_symb in input_symbols :

            if not(self.__is_grammar_token(in_symb)) :
                print(in_symb)
                in_acc = pynini.concat(in_acc, pynini.accep(in_symb, token_type=Sig_tex))
            else :
                for j, out_symb in enumerate(output_symbols[out_idx:]):
                    jk = j + out_idx
                    if not(self.__is_grammar_token(out_symb)):
                        out_acc = pynini.concat(out_acc, pynini.accep(out_symb, token_type=Sig_tex))
                        if jk == len(output_symbols) - 1 : 
                            raise Exception(ValueError("There is more grammar rules in input than in output"))
                    else :
                        if in_symb != out_symb : 
                            raise Exception(ValueError(f"Cannot cross different grammar type : in:{in_symb} and out:{out_symb}"))
                        grammar_acc = self.__get_grammar_acceptor(in_symb)
                        acc = pynini.concat(pynini.concat(acc, pynini.cross(in_acc.optimize(), out_acc.optimize())), grammar_acc)
                        in_acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
                        out_acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
                        out_idx = jk+1
                        break

        if out_idx < len(output_symbols) :
            for out_symb in output_symbols[out_idx:]:
                if not(self.__is_grammar_token(out_symb)):
                    out_acc = pynini.concat(out_acc, pynini.accep(out_symb, token_type=Sig_tex))
                else :
                    raise Exception(ValueError("There is more grammar rules in output than in input"))
            acc = pynini.concat(acc, pynini.cross(in_acc.optimize(), out_acc.optimize()))
        
        self.__fst = acc.optimize()


    def get_fst(self): 
        return self.__fst.set_input_symbols(Sig_tex).set_output_symbols(Sig_tex)

    def accep(self, input_string:str):
        symbols = input_string.split()
        acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
        for s in symbols : 
            acc = pynini.concat(acc, pynini.accep(s, token_type = Sig_tex))
        return acc.optimize()

    def predict(self, input_sentence, clean_space=True):
        acceptor = self.accep(input_sentence)
        res = pynini.shortestpath(acceptor @ self.__fst).paths(output_token_type = Sig_tex).ostring()
        res = res.replace(SPACE_TOKEN, " ")
        if clean_space : res = " ".join(res.split())
        return res
    
    def __is_grammar_token(self, symb):
        return (symb[0] == '[' and symb[-1] == ']') or (symb[0] == '[' and symb[-2] == ']' and symb[-1] in ['*', '+', '-', '>'])
    
    def __get_grammar_acceptor(self, symb):
        symb_cup_list = symb.replace('[', ' ').replace(']',' ').replace('|', ' ').split()
        if symb_cup_list[-1] in ['*', '+']:
            typ = symb_cup_list[-1]
            symb_cup_list = symb_cup_list[:-1]
        else : typ = None
        acc = pynini.Fst()
        for symb in symb_cup_list : 
            if symb[-1] in ['*', '+']:
                sub_typ = symb[-1]
                symb = symb[:-1]
            else : sub_typ = None
            if symb in GRAMMAR_ACC.keys() : sub_acc = GRAMMAR_ACC[symb].copy()
            elif symb in dict(Sig_tex).values() : sub_acc = pynini.accep(symb, token_type=Sig_tex)
            else : 
                raise(Exception(ValueError(f"Unknown grammar type :{symb}")))
            if sub_typ == '+' : sub_acc = pynini.concat(sub_acc, sub_acc.closure())
            elif sub_typ == '*' : sub_acc = sub_acc.closure()
            acc = pynini.union(acc, sub_acc)
        if typ == '+' : return pynini.difference(acc.closure(), pynini.accep(NULL_TOKEN, token_type=Sig_tex)).optimize()
        elif typ == '*' : return acc.closure().optimize()
        return acc.optimize()
    
def get_grammar_acceptor_2(self, symb):
    symb = symb.replace(' ', '')
    symb = symb.replace('[', ' [ ').replace(']', ' ] ').replace('|', ' | ')
    symb = symb.split()
    opens = []
    close = []
    rec_depth = 0
    for i, s in enumerate(symb) : 
        if  s == '[' :
            if rec_depth == 0 :
                opens.append(i)
            rec_depth += 1
        elif s == ']' : 
            rec_depth -= 1
            if rec_depth == 0 :
                if i != len(symb) - 1 and symb[i+1] in ['*', '+', '-', '>'] : close.append(i+1)
                else : close.append(i)
    if len(close) != len(opens) : 
        raise Exception(ValueError(f"Invalid grammar type : {symb}"))
    sub_symb_list = []
    for i in range(len(close)):
        sub_symb_list.append("".join(symb[opens[i]:close[i] + 1] ))
        if i != len(close) - 1 :
            if "|" in symb[close[i]:opens[i+1] + 1] : sub_symb_list.append("|")
            else : sub_symb_list.append(".")
    
    if len(sub_symb_list) > 1 :
        grammar_acc = self.get_grammar_acceptor(sub_symb_list[0])
        for i in range(1, len(sub_symb_list), 2) :
            if sub_symb_list[i] == '|' : grammar_acc = pynini.union(grammar_acc, self.get_grammar_acceptor_2(sub_symb_list[i+1]))
            else : grammar_acc = pynini.concat(grammar_acc, self.get_grammar_acceptor_2(sub_symb_list[i+1]))
    else :
        grammar = sub_symb_list[0]
        if grammar[-1] == ']' :
            sub_grammar = grammar[1:-1]
            typ = None
        elif symb[-1] in ['*', '+', '-', '>'] and symb[-2] == ']':
            sub_symb = grammar[1:-2]
            typ = grammar[-1]
        else : 
            raise Exception(ValueError(f"Invalid grammar type : {symb}"))
        

class GrammarRuleTransducer():

    def __init__(self):
        self.L_star = Sig_tex_acc.closure().optimize()
        self.__R = pynini.Fst()
        self.__fst = pynini.Fst()
        self.__update=True

    def add_grammar_rule(self, fst:pynini.Fst):
        self.__R = pynini.union(self.__R, fst).optimize()
        self.__update = False 

    def get_fst(self):
        if not(self.__update):
            self.__build_fst()
        return self.__fst
    
    def accep(self, input_string:str):
        symbols = input_string.split()
        acc = pynini.accep(NULL_TOKEN, token_type=Sig_tex)
        for s in symbols : 
            acc = pynini.concat(acc, pynini.accep(s, token_type = Sig_tex))
        return acc.optimize()

    def predict(self, input_sentence, clean_space=True):
        acceptor = self.accep(input_sentence)
        res = pynini.shortestpath(acceptor @ self.get_fst()).paths(output_token_type = Sig_tex).ostring()
        res = res.replace(SPACE_TOKEN, " ")
        if clean_space : res = " ".join(res.split())
        return res
    
    def outputs(self, input_sentence, clean_space=True):
        acceptor = self.accep(input_sentence)
        res = list((acceptor @ self.get_fst()).paths(output_token_type = Sig_tex).ostrings())
        if clean_space : res = [" ".join(hyp.split()) for hyp in res]
        return res

    def __build_fst(self):
        D = pynini.determinize(pynini.difference(self.L_star, pynini.concat(self.L_star, pynini.concat(self.__R.copy().project('input'), self.L_star)))).optimize()
        self.__fst = pynini.concat(D, pynini.concat(self.__R, D).closure()).optimize()
        self.__update = True

# *****************************************************************
#
#                          Ftex + Fgram
#
# *****************************************************************

class FullTransducer():
    
    def __init__(self, 
                 math_lex_rule_file=MATH_LEX_KSEQ_RULE_FILE, math_gra_rule_file=MATH_GRA_RULE_RILE, header=True):

        self.lexical_fst = LexMathTransducer(math_rule_file=MATH_LEX_KSEQ_RULE_FILE)
        self.grammar_fst = GrammarRuleTransducer() 

        with open(math_gra_rule_file, 'r', encoding=ENCODING) as gram_file:
            for i, line in enumerate(gram_file):
                if not(header and i == 0) :
                    s = line.split(';')
                    _in, _out = s[0], s[1]
                    self.grammar_fst.add_grammar_rule(GrammarRuleAcceptor(_in, _out).get_fst())
    
    def predict(self, input_sentence:str):
        res = self.lexical_fst.predict(input_sentence)
        res = self.grammar_fst.predict(res)
        return res
    
    def outputs(self, input_sentence:str):
        res = self.lexical_fst.predict(input_sentence)
        res = self.grammar_fst.outputs(res)
        return res
    

