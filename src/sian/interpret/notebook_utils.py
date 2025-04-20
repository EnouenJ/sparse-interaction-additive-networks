import numpy as np

def prettyPrintInteractionPairs(agg,readable_labels,   TOP_J = 100):
    print('Top Accumulated 2D Interactions')
    print('-------------------------------')
    sorted_agg = sorted(agg.keys(), key=lambda k: -agg[k])
    indices2 = []
    for j,key in enumerate(sorted_agg):
        if j<TOP_J:
            print(key[0],key[1])
            print('{:15s}; {:15s}'.format(readable_labels[key[0]],readable_labels[key[1]]))
            print('\t\t',agg[key])#,pairs[key],grads[key])
            print()
        if np.log(agg[key])>-3.5: #TODO: need to accumulate this?
            indices2.append((key[0],key[1]))
    print('indices2',len(indices2))
    print(indices2)


def prettyPrintInteractionSingles(agg,readable_labels,   TOP_J = 100):
    print('Top Accumulated 1D Interactions')
    print('-------------------------------')
    sorted_agg = sorted(agg.keys(), key=lambda k: -agg[k])
    for j,key in enumerate(sorted_agg):
        if j<TOP_J:
            print(key[0])
            print('{:15s};'.format(readable_labels[key[0]]))
            print('\t\t',agg[key])#,pairs[key],grads[key])
            print()




def prettyPrintInteractions(agg, readable_labels, TOP_J = 100):
    print('Top Accumulated kD Interactions')
    print('-------------------------------')
    sorted_agg = sorted(agg.keys(), key=lambda k: -agg[k])
    for j,key in enumerate(sorted_agg):
        if j<TOP_J:
            print(key)
            print(';'.join(["{:15s}".format(readable_labels[i]) for i in key]))
            print('\t\t',agg[key])#,pairs[key],grads[key])
            print()


