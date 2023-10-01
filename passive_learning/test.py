
def test(G, test_subgraphs, sensors, method):
    probs = {u: [] for u in G.nodes}
    correct = {u: [] for u in G.nodes}
    for subgraph in test_subgraphs:
        v = random.choice(list(subgraph.nodes))   # v is considered as the source
        sensor_values = {sensor: 1 if (sensor in subgraph.nodes) and nx.has_path(subgraph, sensor, v)
                         else 0 for sensor in sensors}
        pr, c = prob_of_source_given_test(G, v, sensors, sensor_values, method)
        probs[v] += [pr]
        correct[v] += [c]
    mean_probs = np.mean([np.mean(probs[u]) for u in probs.keys()])
    mean_correct = np.mean([np.mean(correct[u]) for u in correct.keys()])
    print('estimated probability for source: ', mean_probs.round(3),
          '     correctly estimated: ', mean_correct.round(3))


def source_given_test_1(graph, source, sensors, sensor_values):
    result_dict = dict()
    for v in graph.nodes:
        probs_of_sensors_given_v = {sensor: r[sensor, v] if sensor_values[sensor] == 1
                                    else 1 - r[sensor, v] for sensor in sensors}
        prob_of_test_given_v = np.product(list(probs_of_sensors_given_v.values()))
        result_dict[v] = prob_of_test_given_v
    correct = 1 if (max(list(result_dict.values())) == result_dict[source]) else 0
    return result_dict[source] / np.sum(list(result_dict.values())), correct


def source_given_test_2(graph, source, sensors, sensor_values):
    suggested_sources = []
    for v in graph.nodes:
        if all((sensor_values[sensor] == 1) == (r[sensor, v] > alpha[sensor]) for sensor in sensors):
            suggested_sources.append(v)
    if source in suggested_sources and len(suggested_sources) > 0:
        prob = 1 / len(suggested_sources)
        if len(suggested_sources) == 1:
            correct = 1
    return prob, correct


def prob_of_source_given_test(graph, sources, sensors, sensor_values, method):
    if method == 1:
        return source_given_test_1(graph, sources, sensors, sensor_values)
    if method == 2:
        return source_given_test_2(graph, sources, sensors, sensor_values)


for j in range(sub_sample_num):
    subgraph = nx.erdos_renyi_graph(n, pi)
    all_subgraphs.append(subgraph.edge_subgraph(all_edges))


subgraph = nx.Graph()
subgraph.add_nodes_from(range(n))
for e in all_edges:
    if np.random.random() < pi:
        subgraph.add_edge(e[0], e[1])
return subgraph


#suggested_sources = []
# if all((sensor_values[sensor] == 1) == (r[sensor, v] > alpha[sensor]) for sensor in sensors):
#    suggested_sources.append(v)
# for v in c:
#   if v in suggested_sources and len(suggested_sources) > 0:
#      probs[v].append(1 / len(suggested_sources))
#     correct[v].append(1)
# else:
#   probs[v].append(0)
#  correct[v].append(0)

for v in c:
    probs[v].append(sensors_to_source[S_value][v] / np.sum(list(sensors_to_source[S_value].values())))
    correct[v].append(1 if sensors_to_source[S_value][v] == max(list(sensors_to_source[S_value].values())) else 0)


final_probs = [np.mean(probs[u]) for u in probs.keys()]
final_correct = [np.mean(correct[u]) for u in correct.keys()]
print('estimated probability for sources: ', final_probs)
print('correctly estimated: ', final_correct)
mean_probs = np.mean(final_probs)
mean_correct = np.mean(final_correct)
print('estimated probability for source: ', mean_probs,
      '     correctly estimated: ', mean_correct)

probs = {u: [] for u in graph.nodes}
correct = {u: [] for u in graph.nodes}
