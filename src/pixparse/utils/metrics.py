try:
    import Levenshtein
except ImportError:
    Levenshtein = None


def normalized_levenshtein(s1, s2):
    assert Levenshtein is not None, 'please `pip install Levenshtein`'
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]  
        o_q_i = predicted_answers[i]
        max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N
