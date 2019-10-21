# IR2019
>Homework1.2: Ranked retrieval model
![](./report_img/img1.png)

*赵鑫鉴 数据班 201700181053*
## 实验任务
>在Homework1.1的基础上实现最基本的Ranked retrieval model 

• Input：a query (like Ron Weasley birthday) 

• Output: Return the top K (e.g., K = 10) relevant tweets. 

• Use SMART notation: lnc.ltn

• Document: logarithmic tf (l as first character), no idf and cosine normalization 

• Query: logarithmic tf (l in leftmost column), idf (t in second column), no normalization 

• 改进Inverted index，在Dictionary中存储每个term的DF，在posting list中存储term在每个doc中的TF with pairs (docID, tf) 

**本实验中额外实现了lnc.btn / bnc.btn/ lnc.atn / anc.atn**

&emsp;在实验一中实现了倒排索引和布尔查询处理办法，给定一个布尔查询，一篇文档要么满足查询的要求，要么不满足要求。在文档集规模很大的情况下，满足布尔查询的结果文档数量可能 非常多，往往会大大超过用户能够浏览的文档的数目。因此，对搜索引擎来说，对文档进行评 分和排序非常重要。为此，对于给定的查询，搜索引擎会计算每个匹配文档的得分。
![](./report_img/img2.png)
&emsp;一个好的搜索引擎需要快速的返回与输入query相关度高的文档，所以如何衡量相似度是一个很重要的问题。我们在信息检索课程中学习了以下一些衡量相似度的计算方式，并尝试在本次实验中加以实现。

&emsp;如果文档或者域中词项出现的频率越高，那么该文档或者域的得分也越高。这是一种很自然的想法。首先，我们对于词项t，根据其在文档d中的权重来计算它的得分。简单的方式是将权重设置为t在文档中的出现次数。这种权重计算的结果称为词项频率（term frequencey）。同时我们注意到“**Relevance does not increase proportionally with term frequency**”,所以在实际设计中并不是采用这种简单的方法。在本次实验中我们采用的是Log-frequency weighting：
![](./report_img/img3.png)

&emsp;原始的词项频率会面临这样一个严重问题，即在和查询进行相关度计算时，所有的词项都被认为是同等重要的。实际上,某些词项对于相关度计算来说几乎没有或很少有区分能力。 例如，在一个有关汽车工业的文档集中，几乎所有的文档都会包含auto，此时，auto就没有区分能力。为此，需要一种机制来降低这些出现次数过多的词项在相关性计算中的重要性。一个很直接的想法就是给文档集频率（collection frequency）较高的词项赋予较低的权重，其中文档集频率指的是词项在文档集中出现的次数。这样，便可以降低具有较高文档集频率的词项的权重。由于df本身往往较大，所以通常需要将它映射到一个较小的取值范围中去。所以在实际中我们常使用idf。
![](./report_img/img6.png)
![](./report_img/img5.png)

## Requirements
+ python==3.7
+ textblob==0.15.3
+ math
## 实现
1.建立倒排索引表，同时计算两种tf（L和a），idf & cosine normalization。与实验一不同的是postinglists中多存了一个tf。**为了运行高效，将cosine normalization用字典存储，避免了查询时产生多余计算，将时间花在预处理上在工程中应当是有效的。**
```sh
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
        cosin[tweetid] = 0
        MAX = 0
        for te in line:
            x = line.count(te)
            
            MAX = max(MAX, x)
        for te in line1:
            res=line1.count(te)
            
            resc=res#词频
            res=1+math.log10(res)
            
            res1=line1.count(te)
            
            res1=0.5+0.5*res1/MAX
            if te in postings.keys():
                postings[te].append([res,tweetid])#文档中的tf

                postings1[te].append([res1, tweetid])  #文档中的atf

                df[te]=df[te]+1#文档中的idf

                cosin[tweetid]=cosin[tweetid]+resc * resc#文档的词频平方和
            else:
                postings[te] = [[res,tweetid]]

                postings1[te]=[[res1, tweetid]]

                df[te]=1

                cosin[tweetid] = cosin[tweetid]+resc * resc
        #print(postings[te])
    for te in df:
            df[te]=math.log10(30548/df[te])
            
            print(df[te])
    for tw in cosin:
        cosin[tw]=math.sqrt(cosin[tw])
```
2.本次实验添加了RankSearch函数，在其中只需要对句子进行处理即可。直接访问预处理的数据Use SMART notation: lnc.ltn / lnc.btn / bnc.btn / lnc.atn / anc.atn进行评估并排序输出结果：
```sh
def RankSearch():
    str = token(input("Search query >> "))
    choose=input("choose SMART Notations >> ")
    #str是一个句子
    length=len(str)
    str1=set(str)

    print(str1)
    if choose == "lnc.ltn":
        for term in str1:
        #对每个词项，算在句子中的tf和idf
        #print(df[term])
        #print(postings[term])
        #对于有此词项的文档算分
        #idf单词在文档中的出现i
        #print(type(postings[term]))
            res = str.count(term)  # 单词出现的次数
            tf = 1 + math.log10(res)
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
.............................................


  ans = sorted(Q.items(), key=lambda x: x[1], reverse=True)
    i=0
    print("Return the top 10 relevant tweets:")
    while i<10:
        print(ans[i])
        i=i+1
    print("All relevant tweets:")
    print(ans)

```

## 结果展示：

在不同SMART notation输出tweetid和得分的元组，按得分从高到低排序，返回top 10的结果。同时支持返回所有结果。我们可以看到在不同评价指标下top10结果虽然不同，但是相似性还是比较高的：

![](./report_img/img7.png)
![](./report_img/img8.png)
![](./report_img/img9.png)
![](./report_img/img10.png)
![](./report_img/img11.png)
![](./report_img/img12.png)











