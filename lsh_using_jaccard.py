import itertools
import sys
import random
import time
from pyspark import SparkContext

HASHES = 100
BANDS = 50

def conf_spark():
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    return sc

def get_user_mapping(line, users):
    new_line = []
    for _ in line:
        new_line.append(users[_])
    return sorted(new_line)

def generate_hash_params(m):
    a = random.sample(range(1, m), HASHES)
    b = random.sample(range(1, m), HASHES)
    return a,b

def generate_hash(a,b,m, list):
    new_list = []
    for _ in list:
        new_list.append((a * _ + b)%m)
    return new_list

def get_hashed_mapping(m, users_list, count):
    users_list_new = []
    for c in range(count):
        l1 = []
        for _ in range(len(users_list)):
            l1.append(generate_hash(m, users_list[_]))
        users_list_new.append(l1)
    return users_list_new

def find_min(users_list):
    min = sys.maxsize
    for _ in users_list:
        min = _ if _ < min else min
    return min

def split_and_hash(users_hashed_list):
    chunk_lists = list()
    size = int(HASHES / BANDS)
    for index, start in enumerate(range(0, HASHES, size)):
        chunk_lists.append((index, hash(tuple(users_hashed_list[start:start + size]))))
    return chunk_lists

def jaccard_sim(list1, list2):
    num = float(len(set(list1) & set(list2)))
    den = float(len(set(list1) | set(list2)))
    return float(num/den)

def compare_documents(users_business, candidate, threshold):
    sim = jaccard_sim(users_business[candidate[0]], users_business[candidate[1]])
    if sim >= threshold:
        l = list(sorted([candidate[0], candidate[1]]))
        l.append(sim)
        return l

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    sc = conf_spark()
    rdd = sc.textFile(sys.argv[1])
    header = rdd.first()
    rdd = rdd.filter(lambda line: line != header)
    users = rdd.map(lambda line: line.split(",")[0]).distinct().sortBy(lambda line: line).zipWithIndex().collectAsMap()
    users_business = rdd.map(lambda line: (line.split(",")[1], [users[line.split(",")[0]]])).reduceByKey(lambda a,b: list(set(a+b)))

    user_business_dict = users_business.collectAsMap()

    a,b = generate_hash_params(len(users))

    signature_matrix = users_business.map(lambda line: (line[0], [find_min(generate_hash(a[_], b[_], len(users), line[1])) for _ in range(HASHES)]))
    candidates = signature_matrix.flatMap(lambda line: [(_,[line[0]]) for _ in split_and_hash(line[1])]).reduceByKey(lambda a,b: list(set(a+b))).filter(lambda line:len(line[1]) > 1).flatMap(lambda line: itertools.combinations(line[1], 2)).distinct()
    final_candidates = candidates.map(lambda line: compare_documents(user_business_dict, line, 0.5)).filter(lambda line: line != None).sortBy(lambda line: line).collect()
    final_candidates_r_dup = []
    final_candidates_set = set()
    for _ in final_candidates:
        candidate = tuple(_[0:2])
        if candidate not in final_candidates_set:
            final_candidates_set.add(candidate)
            final_candidates_r_dup.append(_)
    with open(sys.argv[2], 'w') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for _ in final_candidates_r_dup:
            f.write(_[0] + "," + _[1] + "," + str(_[2]) + "\n")
    end = time.time() - start
    print(end)