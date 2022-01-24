# 一点点pandas

pandas的数据类型更像是一个数据库或者小小Excel，有很方便的索引

与np高度兼容

## Series、DataFrame

pd的列表是series

DataFrame类似矩阵

### Series 初始化

|符号|运算|
|---|---|
|pd.Series([array or np array])|基本的初始化|
|.date_range('start','end',peroid,freq)|生成日期序列,peroid为长度，freq为周期|

### Dataframe 初始化

|符号|运算|
|---|---|
|.Dataframe(array,index,col,dtype)|生成一个dataframe，索引为index，列名为col，缺省默认1~n|
|.Dataframe({'colName':content,...})|直观地按列生成dataframe|

### Dataframe 成员函数、变量

|符号|运算|
|---|---|
|.dtypes|返回dataframe成员数据类型|
|.colnums||
|.index||
|.describe()|数据特征|
|.transpose()|转置，同时列名和索引也会交换|
|.sort_index(axis,ascending)|按索引排序|
|.soer_values(by)|按照by排序|

### Dataframe 索引/访问

下标形式与成员变量形式均能访问到一个切片；python的 ':' 下标也是允许的

可以按照行、列索引访问

    d['A'];d.A      #按照key访问
    d.loc['']       #按照label/index访问,返回整个向量(列向量)
    d.iloc[]        #按照数字下标访问
    d.ix[]          #混合筛选
    d[logical]      #按逻辑筛选
    
    e.g.
    d.loc[:,['A','B']] #筛选数据：所有行，A、B列
    d[df.A>8]

上述的索引方式可以嵌套使用

    d.A[df.A>4]

### Dataframe 修改与形变、合并

修改：访问到的位置可以直接=赋值

合并：

.concat([A,B,C,...],axis，ignore_index = T/F,join) 合并多个datdframe，ignore_index表示是否重构索引

join表示合并方式

    'outer' 不互通处用nan补
    'inner' 裁剪不同部分



### Dataframe 数据清洗

pd.nan是丢失的数据

|符号|运算|
|---|---|
|.dropna(axis,how)|按axis丢弃含nan数据 ,丢弃法则how = 'any'是存在，'all'是均为nan|
|.fillna(value)|特定值填充nan|
|.isnull()|返回dataframe所有位置是否为null|
|np.any(df.isnull()) == True|运算是否存在null|

### 文件io

pandas支持很多文件的读写

read/.to_xxx

xxx includes:
dddd
|csv|excel|hdf|sql|json|msgpack|html|gdq|stata|sas|clipboard|...|

 