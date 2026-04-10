#least_matching silmilar and most similar non-matching
vector_based_examples_dict_1 = {
    "D2" : [(339, 397), (66, 313)],
    "D3":[(1331, 643), (277, 2817)],
    "D5": [(4324, 2932),(3436, 3626)],
    "D6": [(2770, 752), (1552, 5939)],
    "D7": [(1421, 4027), (5641, 2599)],
    "D8": [(317, 1082), (1813, 1607)]
}

#least_matching silmilar and most similar non-matching
join_examples_dict_1 = {
    "D2" : [(875, 1047), (597, 636)],
    "D3":[(189, 1471), (383, 2775)],
    "D5":[(1528, 1178),(239, 1554)],
    "D6":[(123, 455),(1973, 6473)],
    "D7":[(1041, 1887),(1573, 7192)],
    "D8":[(850, 8443),(1081, 5076)]
}

join_examples_dict_2 = {
    "D2":[(321, 467),(1026, 398)],
    "D3":[(1321, 634), (498, 1911)],
    "D5":[(1206, 1076),(823, 959)],
    "D6":[(3399, 163),(2352, 521)],
    "D7":[(5279, 7786),(3271, 7290)],
    "D8":[(1081, 21587),(2539, 18281)]
}

vector_based_examples_dict_2 = {
    "D2":[(777, 643),(154, 677)],
    "D3":[(1321, 634), (388, 1909)],
    "D5":[(2725, 4694),(4077, 5602)],
    "D6":[(3399, 163),(3269, 5237)],
    "D7":[(4036, 7251),(4986, 7532)],
    "D8":[(215, 6601),(622, 3160)]
}

vector_based_examples_compination = {
    "D2" :  vector_based_examples_dict_1["D2"] + vector_based_examples_dict_2["D2"],
    "D5":  vector_based_examples_dict_1["D5"] + vector_based_examples_dict_2["D5"],
    "D6":  vector_based_examples_dict_1["D6"] + vector_based_examples_dict_2["D6"],
    "D7":  vector_based_examples_dict_1["D7"] + vector_based_examples_dict_2["D7"],
    "D8":  vector_based_examples_dict_1["D8"] + vector_based_examples_dict_2["D8"]
}

vector_join_combination = {
    "D2" :  vector_based_examples_dict_1["D2"] + join_examples_dict_1["D2"],
    "D5":  vector_based_examples_dict_1["D5"] + join_examples_dict_1["D5"],
    "D6":  vector_based_examples_dict_1["D6"] + join_examples_dict_1["D6"],
    "D7":  vector_based_examples_dict_1["D7"] + join_examples_dict_1["D7"],
    "D8":  vector_based_examples_dict_1["D8"] + join_examples_dict_1["D8"]
}

multiple_examples_join = {
    "D2":[(875, 1047), (124, 82), (597, 636), (980, 607)],
}

examples_dict_list = {
    "vector_based_examples_dict_1" : vector_based_examples_dict_1,
    "vector_based_examples_dict_2": vector_based_examples_dict_2,
    "join_examples_dict_1": join_examples_dict_1,
    "join_examples_dict_2": join_examples_dict_2,
    # 'vector_join_combination' : vector_join_combination,
    # "multiple_examples_join" : multiple_examples_join
    # 'vector_based_examples_compination' :  vector_based_examples_compination
}