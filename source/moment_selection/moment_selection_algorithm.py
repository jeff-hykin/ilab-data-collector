import math

# example cases:
#     print(logarithmic_cluster([0,0,0,0.49,0,0,0.5,1,1,1,1,1,1,1,0.44,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0.6,]))     # >>> [(7, 28, 0.64), (32, 33, 0.6)]
#     print(logarithmic_cluster([0,0,0,0.49,0,0,0.5,1,1,1,1,1,1,1,0.44,1,1,0,1,0,1,0,0,1,0,1,1,1,0,0,0,0,0.6]))      # >>> [(7, 33, 0.6169230769230769)]
#     print(logarithmic_cluster([0,0,0,0.49,0,0,0.5,1,1,1,1,1,1,1,0.44,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0.6,0,0,])) # >>> [(7, 28, 0.64)]

def logarithmic_cluster(list_of_confidences, threshold=0.5):
    """
    list_of_confidences: list of values between 0 and 1
    threshold: a value between 0 and 1
    returns: [ (start_index, end_index, confidence), ... ]
    """
    # think of this as getting a group that is on average above the threshold
    # BUT instead of counting new values as equally important
    # they are exponentially important
    # 1,1,1,1,1,1,0,0,0,0,0,0 <- above 0.5 on average
    # 1,1,1,1,1,1,0,0         <- above 0.5 when exponentialy on average
    # 1,1,1,1,1,1,0,0,0,0,0,0 <- NOT above 0.5 when exponentialy on average
    # 
    # then the algorithm has a trimming step where
    # a value like this:
    # 1,1,1,1,1,1,0.51,0,0
    # becomes this:
    # 1,1,1,1,1,1,0.51
    
    def next_step(average, number_in_cluster, remaining_values):
        # if there are no remaining values then stop
        if len(remaining_values) == 0:
            return average, number_in_cluster, remaining_values
        
        # if positive then simply accept
        next_value, *new_remaining_values = remaining_values
        if next_value >= threshold:
            new_number_in_cluster = number_in_cluster + 1
            new_average = (number_in_cluster * average +  next_value)/new_number_in_cluster
            return new_average, new_number_in_cluster, new_remaining_values
        
        # compute the log-discounted average
        new_number_in_cluster = number_in_cluster + 1
        # seriously reduce the strength of large samples
        # (aka exagerate impact of the new value)
        weight = 1 if number_in_cluster == 0 else math.log2(number_in_cluster)
        new_lopsided_average = (weight * average + next_value)/(weight + 1)
        # if still above the threshold, then keep going
        if new_lopsided_average >= threshold:
            new_average = (number_in_cluster * average +  next_value)/new_number_in_cluster
            return new_average, new_number_in_cluster, new_remaining_values
        
        # otherwise fail
        return average, number_in_cluster, remaining_values
    
    def next_group(average, remaining_values):
        number_in_cluster = 0
        original = list(remaining_values)
        original_size = len(remaining_values)
        while True:
            length_before = len(remaining_values)
            average, number_in_cluster, remaining_values = next_step(average, number_in_cluster, remaining_values)
            if length_before <= len(remaining_values):
                break
        group = original[0:(original_size-len(remaining_values))]
        # 
        # trim back (group grabs too much typically, especially long groups)
        # 
        group.reverse()
        amount_to_trim = 0
        for each in group:
            if each >= threshold:
                break
            amount_to_trim += 1
        add_back = group[0:amount_to_trim]
        group = group[amount_to_trim:]
        group.reverse()
        confidence = sum(group)/len(group) if len(group) > 0 else 0
        return average, add_back+remaining_values, group, confidence
    
    # the negative form of the remaining values
    index = 0
    average = 0
    remaining_values = list_of_confidences
    results = []
    while len(remaining_values) > 0:
        # get a group, once the group is found flop back to the other side
        average, remaining_values, group, confidence = next_group(average, remaining_values)
        results.append((index, index+len(group), confidence))
        index += len(group)
        # switch sides
        threshold = 1-threshold
        remaining_values = [ 1-each for each in remaining_values ]
    positive_groups = results[::2]
    # remove 0 length groups
    positive_groups = [ each for each in positive_groups if each[0] != each[1] ]
    
    return positive_groups
