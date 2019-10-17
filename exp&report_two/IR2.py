import re
import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict
import sklearn
import math
import numpy
df = defaultdict(dict)

postings = defaultdict(dict)
print(postings)
f = open(r"C:\Users\86178\Documents\Tencent Files\2683258751\FileRecv\tweets.txt")
lines = f.readlines()  # 读取全部内容

def merge_and(term1, term2):

    answer = []

    if (term1 not in postings) or (term2 not in postings):

        return answer

    else:

        i = len(postings[term1])

        j = len(postings[term2])
#i，j为term1&term2倒排索引表的长度
        x1 = 0

        x2 = 0

        while x1 < i and x2 < j:

            if postings[term1][x1] == postings[term2][x2]:

                answer.append(postings[term1][x1])

                x1 += 1

                x2 += 1

            elif postings[term1][x1] < postings[term2][x2]:

                x1 += 1

            else:

                x2 += 1

        return answer


def merge_or(term1, term2):
    answer = []

    if (term1 not in postings) and (term2 not in postings):

        answer = []

    elif term2 not in postings:

        answer = postings[term1]

    elif term1 not in postings:

        answer = postings[term2]

    else:

        answer = postings[term1]

        for item in postings[term2]:

            if item not in answer:
                answer.append(item)

    return answer


def merge_not(term1, term2):
    answer = []

    if term1 not in postings:

        return answer

    elif term2 not in postings:

        answer = postings[term1]

        return answer

    else:

        ANSWER = postings[term1]

        answer = []

        for ter in ANSWER:

            if ter not in postings[term2]:
                answer.append(ter)

        return answer


def Merge_and(term1, term2, term3):
    Answer = []

    if term3 not in postings:

        return Answer

    else:

        Answer = merge_and(term1, term2)

        if Answer == []:
            return Answer

        ans = []

        i = len(Answer)

        j = len(postings[term3])

        x = 0

        y = 0

        while x < i and y < j:

            if Answer[x] == postings[term3][y]:

                ans.append(Answer[x])

                x += 1

                y += 1

            elif Answer[x] < postings[term3][y]:

                x += 1

            else:

                y += 1

        return ans


def Merge_or(term1, term2, term3):
    Answer = []

    Answer = merge_or(term1, term2);

    if term3 not in postings:

        return Answer

    else:

        if Answer == []:

            Answer = postings[term3]

        else:

            for item in postings[term3]:

                if item not in Answer:
                    Answer.append(item)

        return Answer


def Merge_and_or(term1, term2, term3):
    Answer = []

    Answer = merge_and(term1, term2)

    if term3 not in postings:

        return Answer

    else:

        if Answer == []:

            Answer = postings[term3]

            return Answer

        else:

            for item in postings[term3]:

                if item not in Answer:
                    Answer.append(item)

            return Answer


def Merge_or_and(term1, term2, term3):

    Answer = []

    Answer = merge_or(term1, term2)

    if (term3 not in postings) or (Answer == []):

        return Answer

    else:

        ans = []

        i = len(Answer)

        j = len(postings[term3])

        x = 0

        y = 0

        while x < i and y < j:

            if Answer[x] == postings[term3][y]:

                ans.append(Answer[x])

                x += 1

                y += 1

            elif Answer[x] < postings[term3][y]:

                x += 1

            else:

                y += 1

        return ans


def NaiveSearch(terms):

    Answer = defaultdict(dict)

    for item in terms:

        if item in postings:

            for tweetid in postings[item]:
                print(postings[item])
                if tweetid in Answer:

                    Answer[tweetid] += 1

                else:

                    Answer[tweetid] = 1

    Answer = sorted(Answer.items(), key=lambda asd: asd[1], reverse=True)

    return Answer

Q = defaultdict(dict)
A = defaultdict(dict)


def RankSearch():
    str = token(input("Search query >> "))

    #str是一个句子
    length=len(str)
    str1=set(str)

    print(str1)
    for term in str1:
        #对每个词项，算在句子中的tf和idf
        res=str.count(term)#单词出现的次数

        tf=1+math.log10(res)

        #print(df[term])

        #print(postings[term])
        #对于有此词项的文档算分
        #idf单词在文档中的出现i
        #print(type(postings[term]))

        for te in postings[term]:
            #print(te)
            #print(type(te))
            #print(term)

            tweeid=te[1]
            if A[tweeid]==1:
                Q[tweeid] = Q[tweeid]+tf * df[term] * te[0] / cosin[te[1]]
            else:

             Q[tweeid]=tf*df[term]*te[0]/cosin[te[1]]
             A[tweeid]=1
            #Q[tweeid]=tf*df[term]*te[0]/cosin[te[1]]
    ans = sorted(Q.items(), key=lambda x: x[1], reverse=True)
    i=0
    print("Return the top 10 relevant tweets:")
    while i<10:
        print(ans[i])
        i=i+1
    print("All relevant tweets:")
    print(ans)





def token(doc):
    doc = doc.lower()

    terms = TextBlob(doc).words.singularize()

    result = []

    for word in terms:
        expected_str = Word(word)

        expected_str = expected_str.lemmatize("v")

        result.append(expected_str)

    return result


def tokenize_tweet(document):
    document = document.lower()
    uselessTerm = ["username","clusterno" ,"tweetid","errorcode","text","timestr"]

    a = document.index(uselessTerm[0])

    b = document.index(uselessTerm[1])

    c = document.index(uselessTerm[2])

    d = document.index(uselessTerm[3])

    e = document.index(uselessTerm[4])

    f = document.index(uselessTerm[5])
    # 提取用户名、tweet内容和tweetid三部分主要信息

    document = document[c:d] + document[a:b] + document[e:f]

    # print(document)

    terms = TextBlob(document).words.lemmatize()

    result = []

    for word in terms:

        expected_str = Word(word)

        expected_str = expected_str.lemmatize("v")

        if expected_str not in uselessTerm:
            result.append(expected_str)
    #print(result)
    return result

cosin=dict()
#Document: logarithmic tf (l as first character), no idf and cosine normalization 在某个文档中的词频，正则化，对于倒排索引表，要算词频和正则化
# Query: logarithmic tf (l in leftmost column), idf (t in second column), no normalization 在句子中的词频，在句子中的idf
def get_postings():
    global postings
    global df
    global cosin
    f = open(r"C:\Users\86178\Documents\Tencent Files\2683258751\FileRecv\tweets.txt")
    lines = f.readlines()  # 读取全部内容
    cot=0
    for line in lines:
        cot=cot+1
        line = tokenize_tweet(line)
#list
        tweetid = line[0]
#提取tweetid,并从line中pop
#求cosin，需要每个文档的长度，词频的平方
        line1=line
        line1.pop(0)
        #print(line)
        #unique_terms = set(line1)
        #print(unique_terms)
        #for te in unique_terms:
        #no_use = 0
        cosin[tweetid] = 0
        for te in line1:
            res=line1.count(te)
            resc=res#词频
            #re1=len(line1)
            #res=res/re1
            res=1+math.log10(res)
            if te in postings.keys():
                postings[te].append([res,tweetid])#文档中的tf

                df[te]=df[te]+1#文档中的idf

                cosin[tweetid]=cosin[tweetid]+resc * resc#文档的词频平方和
            else:
                postings[te] = [[res,tweetid]]

                df[te]=1

                cosin[tweetid] = cosin[tweetid]+resc * resc
        #print(postings[te])
    for te in df:
            df[te]=math.log10(30548/df[te])
            print(df[te])
    for tw in cosin:
        cosin[tw]=math.sqrt(cosin[tw])
    #print(cot)

    # postings = sorted(postings.items(),key = lambda asd:asd[0],reverse=False)



def search():
    global line
    terms = token(input("Search query >> "))

    if terms == []:
        sys.exit()

        # 搜索的结果答案

    if len(terms) == 3:

        # A and B

        if terms[1] == "and":

            answer = merge_and(terms[0], terms[2])

            print(answer)

        # A or B

        elif terms[1] == "or":

            answer = merge_or(terms[0], terms[2])

            print(answer)

        # A not B

        elif terms[1] == "not":

            answer = merge_not(terms[0], terms[2])

            print(answer)

        # 输入的三个词格式不对

        else:

            print("input wrong!"+
                  " Please input fomat like 'a and b' or 'a and b or c',and we only supported words<3")


    elif len(terms) == 5:

        # A and B and C

        if (terms[1] == "and") and (terms[3] == "and"):

            answer = Merge_and(terms[0], terms[2], terms[4])

            print(answer)

        # A or B or C

        elif (terms[1] == "or") and (terms[3] == "or"):

            answer = Merge_or(terms[0], terms[2], terms[4])

            print(answer)

        # (A and B) or C

        elif (terms[1] == "and") and (terms[3] == "or"):

            answer = Merge_and_or(terms[0], terms[2], terms[4])

            print(answer)

        # (A or B) and C

        elif (terms[1] == "or") and (terms[3] == "and"):

            answer = Merge_or_and(terms[0], terms[2], terms[4])

            print(answer)

        else:

            print("More format will supported but no now!")

    else:

        leng = len(terms)

        answer = NaiveSearch(terms)

        print("[Rank_Score: Tweetid :Text]")
        ID=[]
        for (tweetid, score) in answer:
            ID.append(tweetid)

        for line in lines:
                line1 = tokenize_tweet(line)
                if line1[0] in ID:
                #从n次优化到一次遍历
                 #print(line)
                 e = line.index("text") +6
                 f = line.index("timeStr") -3
                 w=line[e:f]
                 print(str(score / leng) + ": " + tweetid+":",end="")
                 print(w)


def main():
    get_postings()

    while True:
        #search()
        RankSearch()

if __name__ == "__main__":
    main()