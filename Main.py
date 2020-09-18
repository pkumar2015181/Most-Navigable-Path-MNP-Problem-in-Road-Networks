import xlrd
import sys
from time import perf_counter
import pandas as pd
import random
from random import randint
import os.path
from os import path

node_filename = sys.argv[1]
edge_filename = sys.argv[2]
node_list = []
source_list = []
dest_list = []

def change_to_excel():
    node_file = open(node_filename, "r")
    for x in node_file:
        node_list.append(int(x))

    sorted(node_list)
    print("No. of nodes = ", len(node_list))

    edge_file = open(edge_filename, "r")
    weight_list = []
    for x in edge_file:
        sep = x.split(",")
        source = int(sep[0])
        source_index = node_list.index(source)
        source_list.append(source_index)
        dest = int(sep[1])
        dest_index = node_list.index(dest)
        dest_list.append(dest_index)
        weight_list.append(int(sep[2]))

    print("No. of source = ", len(source_list))
    print("No. of dest = ", len(dest_list))
    print("No. of weights = ", len(weight_list))

    nav_score = [0] * len(source_list)
    print("No. of nav score = ", len(nav_score))

    percent = 40
    no_edge = int((len(source_list) * percent) / 100)
    print("No. of Non-zeroes value in nav score = ", no_edge)
    rand_list = random.sample(range(0, len(source_list)), no_edge)
    for x in rand_list:
        nav_score[x] = random.sample(range(1, 15), 1)[0]

    data = {'source': source_list, 'dest': dest_list, 'weight': weight_list, 'score': nav_score}
    df = pd.DataFrame(data)
    if path.exists("dataset_" +str(len(node_list)) + ".xlsx"): df.to_excel("dataset1.xlsx", index=False, header=False)
    else: df.to_excel("dataset_" +str(len(node_list)) + ".xlsx", index=False, header=False)
    print("output xlsx file is saved")

change_to_excel()
fp = open("Output_" +str(len(node_list)) + "_" + str(sys.argv[3]) + ".txt", "w")

wb = xlrd.open_workbook("dataset_" +str(len(node_list)) + ".xlsx")
sheet = wb.sheet_by_index(0)
no_of_vertex = len(node_list)
adjacency_list = {key: [] for key in range(no_of_vertex)}
edge_list = []
edge_details = []


def add_edge(source, destination, cost, score):
    adjacency_list[int(source)].append([int(destination), cost, score])


def make_edge_list(source, dest, cost, score):
    edge_list.append([int(source), int(dest)])
    edge_details.append([cost, score])


def detect_cycle(path):
    if len(list(set(path))) < len(path): return True
    return False


def cteate_list(dist_queue, parent, lebel, x):
    path = []
    if x== -1:
        ind = randint(0, len(lebel)-1)
        path.append(lebel[ind])
    else: path.append(x)
    while parent[path[-1]] != lebel[0]:
        if path[-1] == -1: return [], 0
        path.append(parent[path[-1]])
    path.append(lebel[0])
    path.reverse()

    return path, dist_queue[path[-1]]


def dijkstra(u, x):
    dist_queue = [sys.maxsize] * no_of_vertex
    parent = [-1] * no_of_vertex
    dist_queue[u] = 0
    p_queue = []
    for i in range(no_of_vertex):
        p_queue.append(i)
    lebel = []
    while p_queue and len(lebel)<50:
        if x==-1 and len(lebel)>=50: break
        d = []
        for i in p_queue:
            d.append(dist_queue[i])
        dest = min(d)
        v = -1
        for i in range(no_of_vertex):
            if dest == dist_queue[i] and p_queue.count(i):
                v = i
                break
        lebel.append(v)
        p_queue.remove(v)
        for i in adjacency_list[v]:
            if p_queue.count(i[0]):
                if dest + i[1] < dist_queue[i[0]]:
                    dist_queue[i[0]] = dest + i[1]
                    parent[i[0]] = v
        if x>-1 and v==x: break
    return cteate_list(dist_queue, parent, lebel, x)



def best_predecessor(ftail, btail, visit_by_backward_search):
    gamma_b = -1
    u = -1
    for i in range(no_of_vertex):
        if i != btail and not visit_by_backward_search[i]:
            for j in adjacency_list[i]:
                if j[0] == btail:
                    path, d = dijkstra(i, j[0])
                    if len(path) != 0:
                        temp = (1 + j[2]) / (j[1] + d)
                        if temp > gamma_b:
                            gamma_b = temp
                            u = i
    return u, gamma_b


def best_successor(ftail, btail, visit_by_forward_search):
    gamma_f = -1
    u = -1
    for i in adjacency_list[ftail]:
        if not visit_by_forward_search[i[0]] and i[0] != btail:
            path, d = dijkstra(i[0], btail)
            if len(path) != 0:
                temp = (1 + i[2]) / (i[1] + d)
                if temp > gamma_f:
                    gamma_f = temp
                    u = i[0]
    return u, gamma_f


def cost(list):
    total_cost = 0
    for i in range(len(list) - 1):
        total_cost += edge_details[edge_list.index([list[i], list[i + 1]])][0]
    return total_cost


def score(list):
    total_score = 0
    for i in range(len(list) - 1):
        total_score += edge_details[edge_list.index([list[i], list[i + 1]])][1]
    return total_score


def check_one_hop_path(ftail_queue, btail_queue, forward_queue, backward_queue):
    if len(list(set(forward_queue) & set(btail_queue))) or len(list(set(backward_queue) & set(ftail_queue))):
        return True
    return False


def check_two_hop_path(forward_queue, backward_queue):
    if len(list(set(forward_queue) & set(backward_queue))):
        return True
    return False


def p_candidate(ftail_queue, btail_queue, node, index):
    tmp_queue1 = []
    if index:
        tmp_queue1.extend(ftail_queue)
        tmp_queue1.extend(btail_queue[btail_queue.index(node) + 1:])
        return tmp_queue1
    else:
        tmp_queue1.extend(ftail_queue[:ftail_queue.index(node)])
        tmp_queue1.extend(btail_queue)
        return tmp_queue1


def p_candidate_one_hop(omega, nav_score, ftail_queue, btail_queue, forward_queue, backward_queue, budget):
    if len(list(set(forward_queue) & set(btail_queue))):
        common_vertex = list(set(forward_queue) & set(btail_queue))
        for j in common_vertex:
            tmp_queue = []
            tmp_queue.extend(ftail_queue)
            tmp_queue.extend(btail_queue[btail_queue.index(j):])
            if cost(tmp_queue) <= budget and not omega.count(tmp_queue):
                omega.append(tmp_queue)
                nav_score.append(score(tmp_queue))
    if len(list(set(backward_queue) & set(ftail_queue))):
        common_vertex = list(set(backward_queue) & set(ftail_queue))
        for j in common_vertex:
            tmp_queue = []
            tmp_queue.extend(ftail_queue[:ftail_queue.index(j) + 1])
            tmp_queue.extend(btail_queue)
            if cost(tmp_queue) <= budget and not omega.count(tmp_queue):
                omega.append(tmp_queue)
                nav_score.append(score(tmp_queue))
    return omega, nav_score


def p_candidate_two_hop(omega, nav_score, ftail_queue, btail_queue, forward_queue, backward_queue, budget):
    if len(list(set(forward_queue) & set(backward_queue))):
        common_vertex = list(set(forward_queue) & set(backward_queue))
        for j in common_vertex:
            tmp_queue = []
            tmp_queue.extend(ftail_queue)
            tmp_queue.append(j)
            tmp_queue.extend(btail_queue)
            if cost(tmp_queue) <= budget and not omega.count(tmp_queue):
                omega.append(tmp_queue)
                nav_score.append(score(tmp_queue))
    return omega, nav_score


def wbs(segment, budget):
    if len(segment) == 0: return segment
    ftail_queue = []
    btail_queue = []
    forward_queue = []
    backward_queue = []
    visit = [0] * no_of_vertex
    ftail = segment[0]
    visit[ftail] = 1
    for i in adjacency_list[ftail]:
        if not visit[i[0]]:
            forward_queue.append(i[0])
    ftail_queue.append(ftail)
    btail = segment[-1]
    visit[btail] = 1
    for i in range(no_of_vertex):
        if i != btail and not visit[i]:
            for j in adjacency_list[i]:
                if j[0] == btail:
                    backward_queue.append(i)
                    break
    btail_queue.append(btail)
    omega = []
    omega.append(segment)
    navigability_score = []
    navigability_score.append(score(segment))
    while not (ftail_queue.count(btail) or btail_queue.count(ftail)):
        flag = 0
        if check_one_hop_path(ftail_queue, btail_queue, forward_queue, backward_queue):  #
            omega, navigability_score = p_candidate_one_hop(omega, navigability_score, ftail_queue, btail_queue,
                                                            forward_queue, backward_queue, budget)
        if check_two_hop_path(forward_queue, backward_queue):  #
            omega, navigability_score = p_candidate_two_hop(omega, navigability_score, ftail_queue, btail_queue,
                                                            forward_queue, backward_queue, budget)
        best_successor_of_ftail, gamma_f = best_successor(ftail, btail, visit)
        best_predecessor_of_btail, gamma_b = best_predecessor(ftail, btail, visit)
        if best_successor_of_ftail > -1 and best_predecessor_of_btail > -1:
            if gamma_f >= gamma_b:
                ftail = best_successor_of_ftail
                forward_queue.clear()
                visit[best_successor_of_ftail] = 1
                for i in adjacency_list[ftail]:
                    if not visit[i[0]]:
                        forward_queue.append(i[0])
                ftail_queue.append(ftail)
                flag = 1
                best_predecessor_of_btail, gamma_b = best_predecessor(ftail, btail, visit)
                if best_predecessor_of_btail != -1:
                    btail = best_predecessor_of_btail
                    backward_queue.clear()
                    visit[best_predecessor_of_btail] = 1
                    for i in range(no_of_vertex):
                        if i != btail and not visit[i]:
                            for j in adjacency_list[i]:
                                if j[0] == btail:
                                    backward_queue.append(i)
                                    break
                    btail_queue.insert(0, btail)
                    flag = 1
            else:
                btail = best_predecessor_of_btail
                backward_queue.clear()
                visit[best_predecessor_of_btail] = 1
                for i in range(no_of_vertex):
                    if i != btail and not visit[i]:
                        for j in adjacency_list[i]:
                            if j[0] == btail:
                                backward_queue.append(i)
                                break
                btail_queue.insert(0, btail)
                flag = 1
                best_successor_of_ftail, gamma_f = best_successor(ftail, btail, visit)
                if best_successor_of_ftail != -1:
                    ftail = best_successor_of_ftail
                    forward_queue.clear()
                    visit[best_successor_of_ftail] = 1
                    for i in adjacency_list[ftail]:
                        if not visit[i[0]]:
                            forward_queue.append(i[0])
                    ftail_queue.append(ftail)
                    flag = 1
        elif best_predecessor_of_btail > -1:
            btail = best_predecessor_of_btail
            backward_queue.clear()
            visit[best_predecessor_of_btail] = 1
            for i in range(no_of_vertex):
                if i != btail and not visit[i]:
                    for j in adjacency_list[i]:
                        if j[0] == btail:
                            backward_queue.append(i)
                            break
            btail_queue.insert(0, btail)
            flag = 1
        elif best_successor_of_ftail > -1:
            ftail = best_successor_of_ftail
            forward_queue.clear()
            visit[best_successor_of_ftail] = 1
            for i in adjacency_list[ftail]:
                if not visit[i[0]]:
                    forward_queue.append(i[0])
            ftail_queue.append(ftail)
            flag = 1
        else:
            break
        ftail = ftail_queue[-1]
        btail = btail_queue[0]
        if cost(ftail_queue) + cost(btail_queue) > budget or not flag:
            break
    if ftail_queue.count(btail) or btail_queue.count(ftail):
        if ftail_queue.count(btail):
            p_cand = p_candidate(ftail_queue, btail_queue, btail, 0)  # 0 for btail present in forward_queue
        if btail_queue.count(ftail):
            p_cand = p_candidate(ftail_queue, btail_queue, ftail, 1)  # 1 for ftail present in backward_queue
        if not omega.count(p_cand):
            omega.append(p_cand)
            navigability_score.append(score(p_cand))
    if len(omega) == 0:
        return omega
    else:
        if detect_cycle(omega[navigability_score.index(max(navigability_score))]):
            return omega[0]
        return omega[navigability_score.index(max(navigability_score))]


def dss(sgain, path, replaced_segment):
    n = len(path)
    for size in range(2, n):
        for i in range(0, n - size):
            j = i + size
            max_score = sgain[i][j]
            index = i
            for k in range(i + 1, j):
                if sgain[i][k] + sgain[k][j] > max_score:
                    max_score = sgain[i][k] + sgain[k][j]
                    index = k
            if index != i:
                temp = []
                if not isinstance (replaced_segment[i][k][0], list):
                    temp.append(replaced_segment[i][k])
                else:
                    temp.extend(replaced_segment[i][k])
                if not isinstance(replaced_segment[k][j][0], list):
                    temp.append(replaced_segment[k][j])
                else:
                    temp.extend(replaced_segment[i][j])
                replaced_segment[i][j] = temp
                sgain[i][j] = max_score
    return replaced_segment[0][n - 1]


def print_mnp(most_navigable_path):
    for i in range(len(most_navigable_path) - 1):
        print(node_list[most_navigable_path[i]], end="->", file=fp)
    print(node_list[most_navigable_path[-1]], file=fp)


def new_overhead(source, overhead, optimal_dss, initial_seed_path):
    path_from_optimal_dss = [source]
    for i in optimal_dss:
        path_from_optimal_dss.extend(i[1:])
    new_overhead = overhead - cost(path_from_optimal_dss) + cost(initial_seed_path)
    return path_from_optimal_dss, new_overhead


def mnp(cost_of_initial_seed_path, initial_seed_path, mnp_score, mnp_dist, replacement_time):
    overhead = int(cost_of_initial_seed_path * float(sys.argv[3])/100)
    start = perf_counter()
    sgain = [[0.0 for i in range(len(initial_seed_path))] for j in range(len(initial_seed_path))]
    replaced_segment = [[0 for i in range(len(initial_seed_path))] for j in range(len(initial_seed_path))]
    for i in range(1, len(initial_seed_path)):
        for j in range(len(initial_seed_path) - i):
            segment_with_most_navigability_score = wbs(initial_seed_path[j:i + 1 + j],
                                                       overhead + cost(initial_seed_path[j:i + 1 + j]))
            if len(segment_with_most_navigability_score) != 0:
                sgain[j][i + j] = score(segment_with_most_navigability_score) - score(initial_seed_path[j:i + 1 + j])
                replaced_segment[j][i + j] = segment_with_most_navigability_score
    end = perf_counter()
    replacement_time +=(end-start)
    optimal_dss = dss(sgain, initial_seed_path, replaced_segment)
    #print(" Time for DSS algo: ", end - start, file=fp)
    if not isinstance(optimal_dss[0], list):
        print("Most navigable path: ", file=fp)
        print_mnp(optimal_dss)
        mnp_score += score(optimal_dss)
        mnp_dist += cost(optimal_dss)
        print("Score gain: " + str(score(optimal_dss) - score(initial_seed_path)) + ", Cost gain: " + str(
            cost(optimal_dss) - cost(initial_seed_path)), file=fp)
        return mnp_score, mnp_dist, replacement_time
    else:
        opt_path, overhead = new_overhead(initial_seed_path[0], overhead, optimal_dss, initial_seed_path)
        output = wbs(opt_path, overhead + cost(opt_path))
        print("Most navigable path: ", file=fp)
        print_mnp(output)
        mnp_score += score(output)
        mnp_dist += cost(output)
        print("Score gain: " + str(score(output) - score(initial_seed_path)) + ", Cost gain: " + str(
            cost(output) - cost(initial_seed_path)), file=fp)
        return mnp_score, mnp_dist, replacement_time


def create_bucket():
    counter1 = counter2 = counter3 = counter4 = 0.0
    bucket = {'11-20': [], '21-30': [], '31-40': [], '>40': []}
    while len(bucket['11-20'])<10 or len(bucket['21-30'])<10 or len(bucket['31-40'])<10 or len(bucket['>40'])<10:
        src = randint(0, len(node_list) - 1)
        start = perf_counter()
        path, cost_path = dijkstra(src, -1)
        end = perf_counter()
        if cost_path>0:
            des = path[-1]
            if cost_path >= 11 and cost_path <= 20 and bucket['11-20'].count((src, des, cost_path, path)) == 0 and len(bucket['11-20'])<10:
                bucket['11-20'].append((src, des, cost_path, path))
                counter1 += (end - start)
            elif cost_path >= 21 and cost_path <= 30 and bucket['21-30'].count((src, des, cost_path, path)) == 0 and len(bucket['21-30'])<10:
                bucket['21-30'].append((src, des, cost_path, path))
                counter2 += (end - start)
            elif cost_path >= 31 and cost_path <= 40 and bucket['31-40'].count((src, des, cost_path, path)) == 0 and len(bucket['31-40'])<10:
                bucket['31-40'].append((src, des, cost_path, path))
                counter3 += (end - start)
            elif cost_path>40 and bucket['>40'].count((src, des, cost_path, path)) == 0 and len(bucket['>40'])<10:
                bucket['>40'].append((src, des, cost_path, path))
                counter4 += (end - start)

    print("Average runtime of Dijkstra for bucket '11-20' in seconds = " + str(counter1/len(bucket['11-20'])), file=fp)
    print("Average runtime of Dijkstra for bucket '21-30' in seconds = " + str(counter2/len(bucket['21-30'])), file=fp)
    print("Average runtime of Dijkstra for bucket '31-40' in seconds = " + str(counter3/len(bucket['31-40'])), file=fp)
    print("Average runtime of Dijkstra for bucket '>40' in seconds = " + str(counter4/len(bucket['>40'])), file=fp)

    print("Size of bucket 11-20 = ", len(bucket['11-20']))
    print("Size of bucket 21-30 = ", len(bucket['21-30']))
    print("Size of bucket 31-40 = ", len(bucket['31-40']))
    print("Size of bucket >40 = ", len(bucket['>40']))
    return bucket


for iterator in range(sheet.nrows):
    add_edge(sheet.cell_value(iterator, 0), sheet.cell_value(iterator, 1), sheet.cell_value(iterator, 2),
             sheet.cell_value(iterator, 3))
    make_edge_list(sheet.cell_value(iterator, 0), sheet.cell_value(iterator, 1), sheet.cell_value(iterator, 2),
                   sheet.cell_value(iterator, 3))

print("Adjacency List created.")
start = perf_counter()
buckets = create_bucket()
end = perf_counter()
print("bucket creation time: ", end - start)

for bucket in buckets:
    time = 0.0
    dijk_score = 0.0
    dijk_dist = 0.0
    replacement_time = 0.0
    mnp_score = 0.0
    mnp_dist = 0.0
    for entry in buckets[bucket]:
        print(bucket)
        print("\n(Source, Destination): ", (node_list[entry[0]], node_list[entry[1]]), file=fp)
        print("Shortest path: ", file=fp)
        print_mnp(entry[3])
        Score = score(entry[3])
        dijk_score += Score
        dijk_dist += entry[2]
        print("Cost = ", entry[2], ", Score = ", Score , file=fp)
        start = perf_counter()
        mnp_score, mnp_dist, replacement_time = mnp(entry[2], entry[3], mnp_score, mnp_dist, replacement_time)
        end = perf_counter()
        time += (end - start)
    avg_time = time / len(buckets[bucket])
    avg_replacement = replacement_time/ len(buckets[bucket])
    print("\n\nAverage runtime of segment replacement for bucket " + str(bucket) + " in seconds = " + str(avg_replacement), file=fp)
    print("Average runtime of DSS + final WBS invokation for bucket " + str(bucket) + " in seconds = " + str(avg_time - avg_replacement), file=fp)
    print("Average runtime for bucket " + str(bucket) + " in seconds = " + str(avg_time), file=fp)
    print("Average score per distance in Dijkstra for bucket " + str(bucket) +" : " + str(dijk_score/dijk_dist), file=fp)
    print("Average score per distance in MNP for bucket " + str(bucket) +" : " + str(mnp_score/mnp_dist) + "\n", file=fp)
fp.close()