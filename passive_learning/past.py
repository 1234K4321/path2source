def test(graph, test_subgraphs, sensors, method):
    global final_acc, final_dist, final_rank
    sensors_to_source = dict()
    dist_list = []
    all_pairs = dict(nx.all_pairs_shortest_path_length(graph))
    rank_list = []
    acc_list = []
    if method == 3:
        sensors_to_source = sensors_to_source_dict(sensors)
    for subgraph in test_subgraphs:
        for c in nx.connected_components(subgraph):
            sensor_values = {sensor: 1 if sensor in c else 0 for sensor in sensors}
            sorted_probs = []

            if method == 1:
                dic = probs_of_test_given_sources(graph, sensors, sensor_values)
                sorted_probs = sorted(graph.nodes, key=lambda u: dic[u] / np.sum(list(dic.values())), reverse=True)

            if method == 2:
                all_probs = dict()
                for v in graph.nodes:
                    pro = 1
                    for sensor in sensors:
                        pro = pro if (sensor_values[sensor] == 1) == (r[sensor, v] > alpha[sensor]) else pro * 0.9
                    all_probs[v] = pro
                sorted_probs = sorted(graph.nodes, key=lambda u: all_probs[u], reverse=True)

            if method == 3:
                S_value = int(''.join('1' if s in c else '0' for s in sensors), 2)
                sorted_probs = sorted(graph.nodes, key=lambda u: sensors_to_source[S_value][u], reverse=True)

            best_estimated = sorted_probs[0]
            # print(len(c))
            # print(best_estimated)
            for v in c:
                acc_list.append(1 if v == best_estimated else 0)
                dist_list.append(all_pairs[best_estimated][v])
                rank_list.append(sorted_probs.index(v) + 1)

    final_acc[method - 1] = list(final_acc[method - 1]) + [mean(acc_list)]
    final_dist[method - 1] = list(final_dist[method - 1]) + [mean(dist_list)]
    final_rank[method - 1] = list(final_rank[method - 1]) + [mean(rank_list)]


