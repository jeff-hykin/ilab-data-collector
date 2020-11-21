def get_moments(list_of_confidences, threshold=0.5):
    # it might be nice to think of this
    # as being similar to a logarithmic kind of K-means
    
    def next_step(average, number_in_cluster, remaining_values):
        # if there are no remaining values then stop
        if len(remaining_values) == 0:
            return average, number_in_cluster, remaining_values
        
        # if positive then simply accept
        next_value, *new_remaining_values = remaining_values
        if next_value > threshold:
            new_number_in_cluster = number_in_cluster + 1
            new_average = (number_in_cluster * average +  next_value)/new_number_in_cluster
            return new_average, new_number_in_cluster, new_remaining_values
        
        # compute the log-discounted average
        new_number_in_cluster = number_in_cluster + 1
        # seriously reduce the strength of large samples
        # (aka exagerate impact of the new value)
        new_lopsided_average = (math.log(number_in_cluster) * average + next_value)/new_number_in_cluster
        # if still above the threshold, then keep going
        if new_lopsided_average > threshold:
            new_average = (number_in_cluster * average +  next_value)/new_number_in_cluster
            return new_average, new_number_in_cluster, new_remaining_values
        
        # otherwise fail
        return average, number_in_cluster, remaining_values
        
    
    # TODO: flip flop between above threshold and below threshold groups
    # TODO: run the sequence backwards once a group-flips-or-flops has been established
    
    # keep calling next_step until it stops shrinking remaining_values
    
    return list_of_observations