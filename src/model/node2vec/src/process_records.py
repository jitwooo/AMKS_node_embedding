if __name__ == "__main__":
    file_path = "/home/master/2020/zhouzhengxiong/workspace/src/common_util/model/node2vec/src/records.txt"

    maxs = {}
    with open(file_path,'r') as f:
        lines = list(filter(lambda x: len(x)!=0,f.readlines()))
        for line in lines:
            name, score = line.strip("#").strip().split(",")
            name, score = name.strip(), float(score.strip())

            if name  in maxs:
                maxs[name] = max(maxs[name],score)
            else:
                maxs[name] = score
    print(maxs)
