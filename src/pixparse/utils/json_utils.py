import re

def json2token(obj, tokenizer_all_special_tokens:list, additional_special_tokens:list=[], update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence and returns the additional json-specific separation tokens.
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys: 
                if update_special_tokens_for_json_key:
                    additional_special_tokens.extend([fr"<s_{k}>", fr"</s_{k}>"])
                jsonentry, additional_special_tokens = json2token(obj[k], tokenizer_all_special_tokens, additional_special_tokens, update_special_tokens_for_json_key, sort_json_key)
                output += (
                    fr"<s_{k}>"
                    + jsonentry
                    + fr"</s_{k}>"
                )
                #self.model.text_decoder.trunk.resize_token_embeddings(
                #    len(self.tokenizer.trunk)
                #    )   
            return output, list(set(additional_special_tokens))
    elif type(obj) == list:
        jsonlist = []
        for item in obj:
            jsonlist_entry, additional_special_tokens = json2token(item, tokenizer_all_special_tokens, additional_special_tokens, update_special_tokens_for_json_key, sort_json_key)
            jsonlist.append(jsonlist_entry)
        return r"<sep/>".join(jsonlist), list(set(additional_special_tokens))
    else:
        obj = str(obj)
        if f"<{obj}/>" in tokenizer_all_special_tokens or f"<{obj}/>" in additional_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj, list(set(additional_special_tokens))