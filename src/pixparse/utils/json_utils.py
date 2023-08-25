import re


def json2token(
    obj,
    tokenizer_all_special_tokens: list,
    additional_special_tokens: list = [],
    update_special_tokens_for_json_key: bool = True,
    sort_json_key: bool = True,
):
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
                    additional_special_tokens.extend([rf"<s_{k}>", rf"</s_{k}>"])
                jsonentry, additional_special_tokens = json2token(
                    obj[k],
                    tokenizer_all_special_tokens,
                    additional_special_tokens,
                    update_special_tokens_for_json_key,
                    sort_json_key,
                )
                output += rf"<s_{k}>" + jsonentry + rf"</s_{k}>"
                # self.model.text_decoder.trunk.resize_token_embeddings(
                #    len(self.tokenizer.trunk)
                #    )
            return output, list(set(additional_special_tokens))
    elif type(obj) == list:
        jsonlist = []
        for item in obj:
            jsonlist_entry, additional_special_tokens = json2token(
                item,
                tokenizer_all_special_tokens,
                additional_special_tokens,
                update_special_tokens_for_json_key,
                sort_json_key,
            )
            jsonlist.append(jsonlist_entry)
        return r"<sep/>".join(jsonlist), list(set(additional_special_tokens))
    else:
        obj = str(obj)
        if (
            f"<{obj}/>" in tokenizer_all_special_tokens
            or f"<{obj}/>" in additional_special_tokens
        ):
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj, list(set(additional_special_tokens))


def token2json(tokens: str, added_vocab=dict(), is_inner_value: bool = False):
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, added_vocab, True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], added_vocab, True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}
