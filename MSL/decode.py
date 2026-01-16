from openpi.models.tokenizer import PaligemmaTokenizer

tokenizer = PaligemmaTokenizer(max_len=200)  # Or whatever max_len you're using
tokens_list = [39635,      1,      1,   3553, 235269,   1104,    603,    476,   2674,   9975, 4018,    575,    573,   5642, 235265,   1165,   8149,    577,    614,    476]
# Decode the tokens - convert JAX array to list first
#tokens_list = text_tokens[0].tolist()  # Get first batch element as Python list
decoded_text = tokenizer._tokenizer.decode(tokens_list)

print(f"Generated text: {decoded_text}")