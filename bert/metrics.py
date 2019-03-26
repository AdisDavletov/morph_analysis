def category_accuracy(gold_df, predicted_df, category='case'):
    gold = gold_df.gram_cats.copy()
    predicted = predicted_df.gram_cats.copy()
    total, correct = 0, 0
    total_sentence, correct_sentence = 0, 0
    for sentence1, sentence2 in zip(gold, predicted):
        flag = True
        for cat1, cat2 in zip(sentence1, sentence2):
            if cat1 == '_' and cat2 == '_':
                continue


            cat1 = cat1.lower()
            cat2 = cat2.lower()

            cat1 = [x for x in cat1.split('|') if x.startswith(category)]
            cat2 = [x for x in cat2.split('|') if x.startswith(category)]

            if len(cat1) == len(cat2):
                if len(cat1) == 0: continue
                correct_cat = cat1[0] == cat2[0]
                if not correct_cat: flag = False
                correct += int(correct_cat)
                total += 1
            else:
                if len(cat1) > 0: flag = False
                total += len(not bool(len(cat1)))
                continue
        total_sentence += 1
        correct_sentence += int(flag)
    correct_sentence *= 100
    correct *= 100
    return correct / total, correct_sentence / total_sentence
