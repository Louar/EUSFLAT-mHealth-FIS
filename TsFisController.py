import pandas as pd
import numpy as np
import json as json

import TsFisService as tfs


# 0) Import data
outputs = ['philanthropist', 'socialiser', 'free_spirit', 'achiever', 'disruptor', 'player']
inputs = [
    'points',           # 1
    'goal_closeness',   # 2
    'activities',       # 3
    'unique_activities',# 4
    'average_activities_per_day', # 5
    'supports',         # 6
    'reactions',        # 7
    'chats',            # 8
    'view_challenge',   # 9
    'view_activities',  # 10
    'view_newsfeed',    # 11
    'view_team',        # 12
    'view_profile',     # 13
    'view_friend',      # 14
]


d = pd.read_csv('./datasets/data-survey-rel.csv', index_col=0)
subjects = d[inputs + outputs]
subjects = subjects.drop([4, 10], axis=0)


# 1) Derive membership functions
memfunc = tfs.deriveMemfunc('./datasets/data-full-rel.csv', inputs)
# print(memfunc)

out = open('./output/out.txt', 'a')
print('% **************', file=out)
print('% **************', file=out)
print('MEMBERSHIP FUNCTIONS', file=out)
out.write(json.dumps(memfunc.to_dict('list'), indent=4))
print('\n\n', file=out)
out.close()



# 2) Derive antecedents
# Derived from expert knowledge
cmd = {
                    #  1   2   3   4   5   6   7   8   9   10  11  12  13  14
    'philanthropist': [0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  1,  0,  1],
    'socialiser':     [0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
    'free_spirit':    [0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    'achiever':       [1,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    'disruptor':      [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    'player':         [0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
}

noa = 14 # Maximum number of antecedents to include in a rule
conditions = ['low', 'medium', 'high'] # The conditions to include in a rule

antecedentsPerOutput = tfs.deriveAntecedents(outputs, inputs, cmd, noa, conditions, memfunc)




subjectsTrain = subjects.sample(frac=0.8, random_state=1)
subjectsTest = subjects.drop(subjectsTrain.index)


# 3) Derive (optimal) consequences per output
rulesPerOutput, ruleMetrics = tfs.deriveRules(memfunc, outputs, antecedentsPerOutput, subjectsTrain);
# rout = open('rules-per-output.json', 'w')
# rout.write(json.dumps(ruleMetrics, indent=4))
# rout.close()


performancePerSubject = tfs.checkPerformance(memfunc, outputs, rulesPerOutput, subjectsTest)
_ = tfs.getPerformanceSummary(outputs, subjectsTest, performancePerSubject, True)









# Perform five k-fold cross-validations
nrOfRepeats, nrOfPartitions = 5, 5
states = []
for n in range(nrOfRepeats):
    if n == 0:
        sse, accOne, accThree = ([] for i in range(3))

    state = np.random.randint(100, size=1)[0]
    while state in states:
        state = np.random.randint(100, size=1)[0]
    states.append(state)

    randIndices = np.random.RandomState(seed = state).permutation(len(subjects))


    out = open('./output/out.txt', 'a')
    print('% **************', file=out)
    print('% **************', file=out)
    print('ITERATION ' + str((n + 1)), file=out)
    print('Random state parameter = ' + str(state) + '\n\n', file=out)
    out.close()

    for k in range(nrOfPartitions):
        testIndices = randIndices[k::nrOfPartitions]

        subjectsTest = subjects.iloc[testIndices]
        subjectsTrain = subjects.drop(subjectsTest.index)

        out = open('./output/out.txt', 'a')
        print('% **************', file=out)
        print('PARTITION ' + str(k + 1), file=out)
        print('Size of train set = ' + str(len(subjectsTrain)), file=out)
        print('Size of test set = ' + str(len(subjectsTest)) + '\n', file=out)
        out.close()


        # 3) Derive (optimal) consequences per output
        rulesPerOutput, ruleMetrics = tfs.deriveRules(memfunc, outputs, antecedentsPerOutput, subjectsTrain)

        out = open('./output/out.txt', 'a')
        out.write(json.dumps(ruleMetrics, indent=4))
        print('\n', file=out)
        out.close()

        rout = open('./output/rules/rules-per-output-' + str(state) + '-' + str(k) + '.json', 'w')
        rout.write(json.dumps(rulesPerOutput, indent=4))
        rout.close()


        # 4) Evaluate performance
        performancePerSubject = tfs.checkPerformance(memfunc, outputs, rulesPerOutput, subjectsTest)

        pout = open('./output/performance/performance-per-subject-' + str(state) + '-' + str(k) + '.json', 'w')
        pout.write(json.dumps(performancePerSubject, indent=4))
        pout.close()



        # Report performance of globally optimized rules
        ssePerOutput, sseTotal, accOneTotal, accThreeTotal = tfs.getPerformanceSummary(outputs, subjectsTest, performancePerSubject)
        out = open('./output/out.txt', 'a')
        print('PERFORMANCE OF GLOBALLY OPTIMIZED RULES', file=out)
        print('Sum of Squared Errors per outcome class = ', file=out)
        out.write(json.dumps(ssePerOutput, indent=4))
        print('\nTotal Sum of Squared Errors = ' + str(sseTotal), file=out)
        print('Accuracy first actual vs first predicted (> 0.167?) = ' + str(accOneTotal), file=out)
        print('Accuracy top three actual vs first predicted (> 0.5?) = ' + str(accThreeTotal) + '\n\n', file=out)
        out.close()

        sse.append(sseTotal)
        accOne.append(accOneTotal)
        accThree.append(accThreeTotal)


    if n == nrOfRepeats - 1:
        out = open('./output/out-validation.txt', 'a')
        print('Validation method: ' + str(nrOfRepeats) + ' times ' + str(nrOfPartitions) + '-fold cross-validation', file=out)
        print('With states: [' + ', '.join(str(state) for state in states) + ']\n', file=out)
        print('PERFORMANCE OF GLOBALLY OPTIMIZED RULES', file=out)
        print('Average total Sum of Squared Errors = ' + str(round(np.mean(sse), 3)) + ', ' + str(round(np.std(sse), 3)), file=out)
        print('Average accuracy first actual vs first predicted (> 0.167?) = ' + str(round(np.mean(accOne), 3)) + ', ' + str(round(np.std(accOne), 3)), file=out)
        print('Average accuracy top three actual vs first predicted (> 0.5?) = ' + str(round(np.mean(accThree), 3)) + ', ' + str(round(np.std(accThree), 3)) + '\n', file=out)
        out.close()
