import re


def clean_name(name):
    name = name.replace('/', '_')
    name = name.replace('-', '_')
    return name


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]
