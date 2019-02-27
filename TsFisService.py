import pandas as pd
import numpy as np
import itertools as iter
import json as json





############################################
## Step 1: Determine membership functions ##
############################################
def deriveMemfunc(data_path, inputs):

    dd = pd.read_csv(data_path, index_col=0)
    ddc = dd[inputs]

    # Derive membership functions from entire dataset
    memfunc  = pd.DataFrame(columns = list(ddc))
    for column in ddc:
        memfunc.at['min', column] = min(ddc[column])
        memfunc.at['median', column] = np.median(ddc[column])
        memfunc.at['max', column] = max(ddc[column])

    return memfunc


def membershipValue(memfunc, x, input, condition):
    min = memfunc.loc['min', input]
    median = memfunc.loc['median', input]
    max = memfunc.loc['max', input]

    if condition.lower() == 'low':
        if x <= min:
            membership = 1
        elif x > min and x <= median:
            membership = (-1 / (median - min)) * x + (median / (median - min))
        else:
            membership = 0

    elif condition.lower() == 'medium':
        if x <= min:
            membership = 0
        elif x > min and x <= median:
            membership = (1 / (median - min)) * x - (min / (median - min))
        elif x > median and x <= max:
            membership = (-1 / (max - median)) * x + (max / (max - median))
        else:
            membership = 0

    elif condition.lower() == 'high':
        if x <= median:
            membership = 0
        elif x > median and x <= max:
            membership = (1 / (max - median)) * x - (median / (max - median))
        else:
            membership = 1

    return membership







############################################
## Step 2: Determine a set of fuzzy rules ##
############################################

def deriveAntecedents(outputs, inputs, cmd, noa, conditions, memfunc):

    # Generate (empty) correlation matrix
    cm = pd.DataFrame(data=cmd, index=inputs)
    acc = cm[cm != 0].stack().reset_index()
    acc.columns = ['input', 'output', 'correlation']

    antecedentsPerOutput = {}
    for output in outputs:
        antecedents = acc[acc.output == output].input.values.tolist()

        # If an output does not have at least noa number of antecedents, by default all inputs that belong to this output are used
        n = min(len(antecedents), noa)

        combinations = list(iter.combinations(antecedents, n))

        conditionsPerInput = []
        for antecedent in antecedents:
            if memfunc[antecedent]['min'] == memfunc[antecedent]['median']:
                conditionsPerInput.append(conditions[1:])
            elif memfunc[antecedent]['min'] == memfunc[antecedent]['median']:
                conditionsPerInput.append(conditions[:-1])
            else:
                conditionsPerInput.append(conditions)

        variations = list(iter.product( *conditionsPerInput ))

        for combination in combinations:
            for variant in variations:
                antecedents = ''
                for i, input in enumerate(combination):
                    # Determine an antecedent
                    antecedent = '\'' + input + '\' = ' + variant[i]

                    if len(antecedents) != 0: antecedents += ' and '
                    antecedents += antecedent

                if output not in antecedentsPerOutput:
                    antecedentsPerOutput[output] = []
                antecedentsPerOutput[output].append(antecedents)

    return antecedentsPerOutput











def deriveConsequences(memfunc, output, antecedentsPerOutput, subjects):

    ruleActionsPerRule = [] # list with {inputs: ..., operators: ..., outputs: ..., values: ...},
    antecedentsPerRule = [] # list of optimal antecedents, i.e. potential rules that never fire for this set of subjects are removed
    consequVarsPerRule = [] # list of optimal consequence input variables, i.e. potential rules that never fire for this set of subjects are removed
    fulfillmentPerSubject = {} # list of optimal consequence input variables, i.e. potential rules that never fire for this set of subjects are removed
    for i, antecedents in enumerate(antecedentsPerOutput):

        ruleActions = []
        for uid, subject in subjects.iterrows():

            degreeOfFulfillment = getDegreeOfFulfillment(memfunc, antecedents, subject)

            ruleAction = {'rule': i, 'subject': uid, 'outputs': [], 'values': []}
            ruleAction['outputs'].append('error_term')
            ruleAction['values'].append(1 * degreeOfFulfillment) # Include the error term

            for antecedent in antecedents.split(' and '): # Include the other coefficients
                input = antecedent.split(' ')[0].replace('\'','') # get the name of the `input variable`
                ruleAction['outputs'].append(input)
                ruleAction['values'].append(subject[input] * degreeOfFulfillment)

            ruleActions.append(ruleAction)

        if sum([ruleAction['values'][0] for ruleAction in ruleActions]) > 0:
            # Delete this potential rule if degreeOfFulfillment == 0 for all subjects
            ruleActionsPerRule += ruleActions

            antecedentsPerRule.append(antecedents)
            consequVarsPerRule.append(ruleAction['outputs'])

            for ruleAction in ruleActions:
                if ruleAction['subject'] not in fulfillmentPerSubject:
                    fulfillmentPerSubject[ruleAction['subject']] = 0
                fulfillmentPerSubject[ruleAction['subject']] += ruleAction['values'][0]

    # Correct the values for one's total degree of fulfillment
    for i, ruleAction in enumerate(ruleActionsPerRule):
        uid = ruleAction['subject']
        if fulfillmentPerSubject[uid] > 0:
            ruleAction['values'] = [x / fulfillmentPerSubject[uid] for x in ruleAction['values']]
            ruleActionsPerRule[i] = ruleAction

    ruleActionsPerSubjects = []
    for uid, subject in subjects.iterrows():
        ruleActionsPerSubjectRaw = list(iter.compress(ruleActionsPerRule, [ruleAction['subject'] == uid for ruleAction in ruleActionsPerRule]))
        ruleActionsPerSubject = list(iter.chain.from_iterable( [ruleAction['values'] for ruleAction in ruleActionsPerSubjectRaw] ))
        ruleActionsPerSubjects.append( {'subject': uid, 'actionsPerRule': ruleActionsPerSubject} )


    # Derive optimal consequent parameters
    actualRuleActionsPerSubject = [ruleActionsPerSubject['actionsPerRule'] for ruleActionsPerSubject in ruleActionsPerSubjects]

    actualOutputs = []
    for ruleAction in ruleActionsPerSubjects:
        subject = ruleAction['subject']
        actualOutputs.append(subjects.loc[subject][output])

    betas = np.linalg.pinv(np.array(actualRuleActionsPerSubject)) @ np.array(actualOutputs) # @ stands for matrix production

    rules = []
    n = 0
    for i, antecedents in enumerate(antecedentsPerRule):
        consequenceVars = consequVarsPerRule[i]

        consequence = ''
        for j, consequenceVar in enumerate(consequenceVars):
            if betas[n + j] != 0:
                if consequenceVar == 'error_term':
                    consequence += str(betas[n + j])
                else:
                    consequence += ' + ' + str(betas[n + j]) + ' * \'' + str(consequenceVar) + '\''

        rules.append((antecedents, consequence))
        n += len(consequenceVars)


    return rules











def deriveRules(memfunc, outputs, antecedentsPerOutput, subjects):
    rulesPerOutput = {}
    ruleMetrics = {}
    for output in outputs:

        # Global optimization of rules
        rulesPerOutput[output] = deriveConsequences(memfunc, output, antecedentsPerOutput[output], subjects)

        ruleMetrics[output] = {
            'nrOfPotentialRules': len(antecedentsPerOutput[output]),
            'nrOfRules': len(rulesPerOutput[output]),
        }

    return rulesPerOutput, ruleMetrics









def checkPerformance(memfunc, outputs, rulesPerOutput, subjects):
    performancePerSubject = {}
    for output in outputs:
        for i, subject in subjects.iterrows():
            prediction = makePrediction(memfunc, subject, rulesPerOutput[output])

            if i not in performancePerSubject:
                performancePerSubject[i] = {}

            if output not in performancePerSubject[i]:
                performancePerSubject[i][output] = {}

            performancePerSubject[i][output] = {
                'actual': subject[output],
                'predicted': prediction,
                'squaredError': (subject[output] - prediction) ** 2
            }

    return performancePerSubject



def getPerformanceSummary(outputs, subjects, performancePerSubject, printResults = False):

    # Sum Squared Error
    ssePerOutput = {}
    for output in outputs:
        sse = sum([performancePerSubject[subject][output]['squaredError'] for subject in performancePerSubject])
        ssePerOutput[output] = sse

    sseTotal = sum(ssePerOutput[performance] for performance in ssePerOutput)

    # Accuracy
    accOne = 0
    accThree = 0
    for uid in performancePerSubject:
        subject = subjects[subjects.index == uid]

        actuals = subject[outputs].sort_values((subject[outputs] * -1).last_valid_index(), axis=1).columns.tolist()[::-1]
        predicted = max(performancePerSubject[uid], key=(lambda key: performancePerSubject[uid][key]['predicted']))

        if predicted == actuals[0]:
            accOne += 1
        if predicted in actuals[0:3]:
            accThree += 1

    accOneTotal = accOne / len(subjects) # Accuracy first actual vs first predicted (> 0.167?)
    accThreeTotal = accThree / len(subjects) # Accuracy top three actual vs first predicted (> 0.5?)


    if printResults:
        print('Sum of Squared Errors per outcome class = ')
        print(json.dumps(ssePerOutput, indent=4) + '\n')
        print('Total Sum of Squared Errors = ' + str(sseTotal))
        print('Accuracy first actual vs first predicted (> 0.167?) = ' + str(accOneTotal))
        print('Accuracy top three actual vs first predicted (> 0.5?) = ' + str(accThreeTotal))

    return ssePerOutput, sseTotal, accOneTotal, accThreeTotal





def makePrediction(memfunc, subject, rules):
    actions = 0
    degreeOfFulfillments = 0

    for rule in rules:
        antecedents = rule[0]
        consequences = rule[1]

        degreeOfFulfillment = getDegreeOfFulfillment(memfunc, antecedents, subject)

        if degreeOfFulfillment > 0:
            for antecedent in antecedents.split(' and '):
                input = antecedent.split(' ')[0].replace('\'','')
                consequences = consequences.replace('\'' + input + '\'', str(subject[input]))

            action = eval(consequences)

            actions += degreeOfFulfillment * action
            degreeOfFulfillments += degreeOfFulfillment

    if degreeOfFulfillments > 0:
        return actions / degreeOfFulfillments
    else:
        return 0










def getDegreeOfFulfillment(memfunc, antecedents, subject):
    mvs = [] # List of membership values per input parameters

    antecedents = antecedents.split(' and ')
    for antecedent in antecedents:
        antel = antecedent.split(' ');

        input = antel[0].replace('\'','') # get the name of the input variable
        condition = antel[2] # get the condition of the input variable

        mv = membershipValue(memfunc, subject[input], input, condition)
        mvs.append(mv)

    # Apply Zadeh's AND operator
    # NOTE: currently, only Zadeh's AND operator is supported,
    # since - for now - all antecedents are combined using this operator
    degreeOfFulfillment = min(mvs)

    return degreeOfFulfillment
