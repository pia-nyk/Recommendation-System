import time
import math
from pyspark import SparkContext
import sys
import task2_2_try
import json
from datetime import datetime

ALPHA = 0.9

def conf_spark():
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    return sc

def calculate_similarity(pair):
    ratings_for_p1 = business_user_ratings_dict.value[pair[0]]
    ratings_for_p2 = business_user_ratings_dict.value[pair[1]]
    corated = set(ratings_for_p1.keys() & ratings_for_p2.keys())

    num = 0.0
    sum1 = 0.0
    sum2 = 0.0

    for user in corated:
        num += ((ratings_for_p1[user]) * (ratings_for_p2[user]))
        sum1 += (math.pow(ratings_for_p1[user], 2))
        sum2 += (math.pow(ratings_for_p2[user], 2))
    if sum1 == 0 or sum2 == 0:
        return 3.0
    return num / math.sqrt(sum1 * sum2)


def similarity_scores(line):
    item = line[0][1]
    user = line[0][0]
    all_eligible_items = line[1]

    scores = {}
    for _ in all_eligible_items:
        pair = tuple(sorted([_, item]))
        score = calculate_similarity(pair)
        if score > 0.0:
            scores[_] = score

    scores_tup = sorted(scores.items(), key=lambda x: -x[1])
    return scores_tup


def predict(line, similarity_scores_list):
    user = line[0]

    num = 0.0
    den = 0.0
    for _ in similarity_scores_list:
        nb = _[0]
        score = _[1]
        num += (score * business_user_ratings_dict.value[nb][user])
        den += abs(score)

    if den == 0:
        return 0.0
    return num / den


def predict_unknown(line):
    user = line[0]
    item = line[1]

    prediction = 0.0
    if user not in users_map_bc.value:
        item_ratings = business_user_ratings_dict.value[items_map_bc.value[item]].values()
        prediction = sum(item_ratings) / len(item_ratings)
    else:
        user_ratings = user_business_ratings_dict.value[users_map_bc.value[user]].values()
        prediction = sum(user_ratings) / len(user_ratings)
    return prediction


def preprocess_data(train):
    # get all the distinct items and index the items
    items_map = train.map(lambda line: line[0]).distinct().zipWithIndex()
    inverted_items_map = items_map.map(lambda line: (line[1], line[0])).collectAsMap()
    # total number of distinct businesses
    total_items = items_map.count()
    items_map = items_map.collectAsMap()

    # get all distinct users and index the users
    users_map = train.map(lambda line: line[1]).distinct().zipWithIndex()
    inverted_users_map = users_map.map(lambda line: (line[1], line[0])).collectAsMap()
    users_map = users_map.collectAsMap()
    return inverted_items_map, total_items, items_map, inverted_users_map, users_map


def convert_strs_to_int(train):
    return train.map(lambda line: (line[0], (line[1], line[2]))).map(
        lambda line: (items_map[line[0]], (users_map[line[1][0]], line[1][1])))


def get_user_business_ratings_dict(mappedRDD):
    return mappedRDD.map(lambda line: (line[1][0], (line[0], line[1][1]))).groupByKey().mapValues(dict).collectAsMap()


def get_ratings(mappedRDD):
    # get all businesses rated by a user
    user_business_dict = mappedRDD.map(lambda line: (line[1][0], [line[0]])).reduceByKey(
        lambda a, b: list(set(a + b))).collectAsMap()
    business_user_ratings_dict = mappedRDD.map(lambda line: (line[0], [line[1]])).reduceByKey(
        lambda a, b: list(set(a + b))).mapValues(dict).collectAsMap()
    return user_business_dict, business_user_ratings_dict


def get_predictions(validation):
    # get predictions for all pairs with unseen item or user
    unknown = validation.filter(lambda line: line[0] not in users_map or line[1] not in items_map).map(
        lambda line: (line, predict_unknown(line))).collectAsMap()

    # get predictions for all pairs with seen items and users
    known = validation.filter(lambda line: line[0] in users_map and line[1] in items_map).map(
        lambda line: (users_map[line[0]], items_map[line[1]])) \
        .map(lambda line: (line, user_business_dict_bc.value[line[0]])).map(
        lambda line: (line[0], similarity_scores(line))).map(
        lambda line: (line[0], predict(line[0], line[1]))).collectAsMap()
    return unknown, known


if __name__ == '__main__':
    start = time.time()
    # file paths
    train_file = sys.argv[1] + "/" + "yelp_train.csv"
    validation_file = sys.argv[2]
    # test_file = sys.argv[3]

    sc = conf_spark()

    # get the training file and preprocess it
    train = sc.textFile(train_file)
    train_headers = train.first()
    train = train.filter(lambda line: line != train_headers).map(
        lambda line: (line.split(",")[1], line.split(",")[0], float(line.split(",")[2])))

    inverted_items_map, total_items, items_map, inverted_users_map, users_map = preprocess_data(train)
    items_map_bc = sc.broadcast(items_map)
    users_map_bc = sc.broadcast(users_map)

    # convert the strings from training data to indexes
    mappedRDD = convert_strs_to_int(train)

    user_business_ratings_dict = sc.broadcast(get_user_business_ratings_dict(mappedRDD))

    user_business_dict, business_user_ratings_dict_rw = get_ratings(mappedRDD)
    user_business_dict_bc = sc.broadcast(user_business_dict)
    # get business-user-ratings dict
    business_user_ratings_dict = sc.broadcast(business_user_ratings_dict_rw)

    # validation file
    validation = sc.textFile(validation_file)
    v_headers = validation.first()
    validation = validation.filter(lambda line: line != v_headers).map(
        lambda line: (line.split(",")[0], line.split(",")[1]))
    validation_model = validation.map(lambda line: (line[1], line[0]))

    validation_ans = sc.textFile(validation_file).filter(lambda line: line != v_headers).map(lambda line: ((line.split(",")[0], line.split(",")[1]), line.split(",")[2])).collectAsMap()
    # print(validation_ans)

    unknown, known = get_predictions(validation)

    # business
    business = sc.textFile(sys.argv[1] + "/" + "business.json").map(json.loads)

    cities = business.map(lambda line: line['city']).zipWithIndex().collectAsMap()

    states = business.map(lambda line: line['state']).zipWithIndex().collectAsMap()

    business = business.map(lambda line: (
    line['business_id'], (cities[line['city']], states[line['state']], line['stars'], line['review_count'])))

    today = datetime.today()
    # user
    user = sc.textFile(sys.argv[1] + "/" + "user.json").map(json.loads).map(lambda line: (line['user_id'], (
    line['review_count'], (today.year - datetime.strptime(line['yelping_since'], '%Y-%m-%d').year) * 12 + (
                today.month - datetime.strptime(line['yelping_since'], '%Y-%m-%d').month), line['average_stars'],line['fans'])))

    model = task2_2_try.train_model(train.map(lambda line: (line[0], (line[1], line[2]))), business, user)
    preds, user_business_cols = task2_2_try.predict(model, validation_model, business, user)
    model_map = {}
    for _ in range(len(preds)):
        model_map[(user_business_cols['User Id'][_],user_business_cols['Business Id'][_])] = preds[_]

    f = open(sys.argv[3], 'w')
    f.write("user_id, business_id, prediction \n")

    rmse = 0.0
    for k, v in known.items():
        user = inverted_users_map[k[0]]
        item = inverted_items_map[k[1]]
        pred1 = v
        pred2 = model_map[(user, item)]
        pred = float((v * (1-ALPHA)) + (ALPHA * model_map[(user, item)]))
        real = float(validation_ans[(user, item)])
        f.write(user + "," + item + "," + str(pred) + "\n")
        rmse += math.pow(pred - real, 2)

    # print("======================== Unknown values =============================")
    for k, v in unknown.items():
        pred1 = v
        pred2 = model_map[(k[0], k[1])]
        pred = float((v * (1-ALPHA)) + (ALPHA * model_map[(k[0], k[1])]))
        real = float(validation_ans[(user,item)])
        f.write(k[0] + "," + k[1] + "," + str(pred) + "\n")
        rmse += math.pow(pred - real, 2)

    rmse /= (len(known) + len(unknown))
    rmse = math.sqrt(rmse)
    print(rmse)

    end = time.time() - start
    print(end)