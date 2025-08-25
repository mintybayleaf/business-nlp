def hamming_distance(one, two):
   if len(one) != len(two):
      return 0
   else:
      return sum([two[pos] == character for pos, character in enumerate(one)])

def jaccard_similarity(tokens_one, tokens_two):
   return len(set(tokens_one) & set(tokens_two)) / len(set(tokens_one) | set(tokens_two))

