import re


class SmilesTokenizer:

    def __init__(self):

        self.PAD = "<PAD>"
        self.START = "<START>"
        self.END = "<END>"
        self.UNK = "<UNK>"

        special = [
            self.PAD,
            self.START,
            self.END,
            self.UNK
        ]

        smiles_tokens = [

            "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si",

            "b", "c", "n", "o", "p", "s",

            "-", "=", "#",
            ":", "/", "\\", ".",

            "(", ")", "[", "]",

            "@", "@@",

            "+", "-",

            "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9",

            "*",

            "%",

            "[O-]",
            "[NH+]",
            "[NH2+]",
            "[NH3+]",
            "[N+]",
            "[nH]",
            "[C@H]",
            "[C@@H]",
            "[O+]",
            "[S-]",
            "[P+]"
        ]

        self.tokens = special + smiles_tokens

        self.stoi = {
            tok: idx
            for idx, tok in enumerate(self.tokens)
        }

        self.itos = {
            idx: tok
            for tok, idx in self.stoi.items()
        }

        self.pattern = re.compile(
            r"\[[^\]]+\]|"
            r"Cl|Br|Si|"
            r"@@?|"
            r"\%\d{2}|"
            r"\d|"
            r"\*|"
            r"[A-Z][a-z]?|"
            r"[bcnohps]|"
            r"[\(\)\[\]\=\#\-\+\:\\\/\.]"
        )

    def vocab_size(self):
        return len(self.tokens)

    def tokenize(self, smiles):
        return self.pattern.findall(smiles)

    def encode(self, smiles, max_len=256):

        tokens = self.tokenize(smiles)

        ids = [self.stoi[self.START]]

        for token in tokens:

            ids.append(
                self.stoi.get(
                    token,
                    self.stoi[self.UNK]
                )
            )

        ids.append(self.stoi[self.END])

        if len(ids) < max_len:

            ids += [self.stoi[self.PAD]] * (
                max_len - len(ids)
            )

        else:

            ids = ids[:max_len]
            ids[-1] = self.stoi[self.END]

        return ids

    def decode(self, token_ids):

        tokens = []

        for idx in token_ids:

            token = self.itos.get(
                int(idx),
                self.UNK
            )

            if token in [
                self.PAD,
                self.START
            ]:
                continue

            if token == self.END:
                break

            tokens.append(token)

        return "".join(tokens)