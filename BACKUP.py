import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sqlite3

files = glob('yob*')
files.sort()

#TODO ZAD 1-3
def Zad123():
    #TODO ZAD 1
    data = pd.concat((pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"]) for file in files))
    #print(data)

    #TODO ZAD 2.
    df2 = data.groupby('name').nunique()
    #print(df2)
    print("Zad2. Nadano 99444 unikalne imiona.")

    #TODO ZAD 3
    df3 = data.groupby('sex')['name'].nunique()
    #print(df3)
    print("Zad3. Nadano 68332 unikalne imiona dla dziewczynek i 42054 dla chlopcow.")

#TODO ZAD 4
def Zad4():
    series_list = []
    dfs=[]
    for file in files:
        df4 = pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"])
        Sexes = df4.groupby('sex')['number'].sum().reset_index()
        series_list.append(Sexes)
        df4.loc[(df4['sex']=='F'), 'frequency_female'] = df4['number']/Sexes.iloc[0,1]
        df4.loc[(df4['sex']=='M'), 'frequency_male'] = df4['number']/Sexes.iloc[1,1]
        dfs.append(df4)

    df4=pd.concat(dfs)
    print("Zad4. Czestotliwosc imion: ")
    print(df4)

#TODO ZAD 5
def Zad5():
    list=[]
    both_list=[]
    femmale = []
    for file in files:
        df5 = pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"])
        Sexes = df5.groupby('sex')['number'].sum().reset_index()
        Both = Sexes.iloc[0,1]+Sexes.iloc[1,1]
        list.append(Sexes)
        both_list.append(Both)
        years = np.arange(1880, 2020, 1).tolist()
        fem2male = Sexes.iloc[0,1]/Sexes.iloc[1,1]
        femmale.append(fem2male)

    dict = {'year':years,'born':both_list, 'count':femmale}
    df = pd.DataFrame(dict)

    print("Zad5. Max")
    print(df.loc[df['count'].idxmax()])
    print("Zad5. Min")
    print(df.loc[df['count'].idxmin()])

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(years,both_list)
    ax2.plot(years,femmale)
    plt.show()

#TODO ZAD 6
def Zad6():
    list = []
    fem = []
    male = []
    years = np.arange(1880, 2020, 1).tolist()
    for file in files:
        df = pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"])
        names = df.groupby('name')['number'].sum().reset_index()
        list.append(names)
        females = df.loc[(df['sex']=='F')]
        males = df.loc[(df['sex']=='M')]
        top_female = females.nlargest(1000,['number'])
        top_male = males.nlargest(1000,['number'])
        fem.append(top_female)
        male.append(top_male)

    #Top 1000 dla kobiet dla kazdego roku
    #print(fem)
    #Top 1000 dla mezczyzn dla kazdego roku
    #print(male)

    data_fem = pd.concat(fem)
    data_fem = data_fem.groupby(['name']).agg({'number':'sum'}).reset_index()
    top_females = data_fem.nlargest(1000, ['number'])
    data_male = pd.concat(male)
    data_male = data_male.groupby(['name']).agg({'number': 'sum'}).reset_index()
    top_males = data_male.nlargest(1000, ['number'])

    #Ogolny ranking

    print(top_females)
    print(top_males)

    print("Zad6. Najpopularniejsze zenskie imie - Mary(4128052)\nNajpopularniejsze meskie imie - James(5177716)")

#TODO ZAD 7
def Zad7():
    fem = []
    male = []
    marilin = []
    harry = []
    james = []
    mary = []
    years = np.arange(1880, 2020, 1).tolist()
    counter = 1879
    tab=[]
    series_list=[]
    m=[]
    h=[]
    j=[]
    mar=[]
    for file in files:
        df = pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"])
        females = df.loc[(df['sex'] == 'F')]
        males = df.loc[(df['sex'] == 'M')]
        top_female = females.nlargest(1000, ['number'])
        top_male = males.nlargest(1000, ['number'])
        fem.append(top_female)
        male.append(top_male)

        Marilin = df.loc[df['name'] == 'Marilin']
        Harry = df.loc[df['name'] == 'Harry']
        James = df.loc[df['name'] == 'James']
        Mary = df.loc[df['name'] == 'Mary']

        df2 = pd.pivot_table(Harry, index=["name"])
        df4 = pd.pivot_table(James, index=["name"])
        df5 = pd.pivot_table(Mary, index=["name"])

        if (Marilin.shape[0]==0):
            counter+=1
        if(Marilin.shape[0]==1):
            counter+=1
            tab.append(counter)

        marilin.append(Marilin.iloc[:,2])
        harry.append(df2.iloc[0,0])
        james.append(df4.iloc[0,0])
        mary.append(df5.iloc[0,0])

        Sexes = df.groupby('sex')['number'].sum().reset_index()
        series_list.append(Sexes)
        df.loc[(df['sex'] == 'F'), 'frequency_female'] = df['number'] / Sexes.iloc[0, 1]
        df.loc[(df['sex'] == 'M'), 'frequency_male'] = df['number'] / Sexes.iloc[1, 1]

        M = df.loc[df['name'] == 'Marilin']
        H = df.loc[df['name'] == 'Harry']
        J = df.loc[df['name'] == 'James']
        Mar = df.loc[df['name'] == 'Mary']

        popmar = pd.pivot_table(Mar, index=["number"])
        poph = pd.pivot_table(H, index=["number"])
        popj = pd.pivot_table(J, index=["number"])

        h.append(poph.iloc[0, 0])
        j.append(popj.iloc[0, 0])
        mar.append(popmar.iloc[0, 0])

    marilin = pd.concat(marilin)

    plt.subplot(121)
    plt.plot(tab, marilin)
    plt.plot(years, harry)
    plt.plot(years, james)
    plt.plot(years, mary)
    plt.subplot(122)
    #plt.plot(years, m)
    plt.plot(years, h)
    plt.plot(years, j)
    plt.plot(years, mar)
    plt.show()

#TODO ZAD 8
def Zad8():
    series_list = []
    fem = []
    male=[]
    ALL=[]
    tab1=[]
    tab2=[]
    years = np.arange(1880, 2020, 1).tolist()
    for file in files:
        df = pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"])
        Sexes = df.groupby('sex')['number'].sum().reset_index()
        series_list.append(Sexes)
        females = df.loc[(df['sex'] == 'F')]
        males = df.loc[(df['sex'] == 'M')]
        tab1.append(Sexes.iloc[0,1])
        tab2.append(Sexes.iloc[1,1])
        top_female = females.nlargest(1000, ['number'])
        top_male = males.nlargest(1000, ['number'])
        F = top_female.groupby('sex')['number'].sum().reset_index()
        M = top_male.groupby('sex')['number'].sum().reset_index()

        fem.append(F.iloc[0,1])
        male.append(M.iloc[0,1])

    ALL=[x + y for x, y in zip(tab1, tab2)]
    per_fem=[x/y for x,y in zip(fem,ALL)]
    per_male=[x/y for x,y in zip(male,ALL)]
    disp=[x-y for x,y in zip(fem,male)]

    counter=1879
    for i in disp:
        max=0
        min=0
        counter+=1
        if(i>max):
            max=i

    plt.plot(years, per_fem)
    plt.plot(years, per_male)
    plt.show()

#TODO ZAD 10
def Zad10():
    counter=0
    list=[]
    data = pd.concat((pd.read_csv(file, sep=",", header=None, names=["name", "sex", "number"]) for file in files))
    group = data.groupby('sex')['name'].unique()
    df3 = group[group.apply(lambda x: len(x)>1)]
    Females = df3.iloc[0]
    Males = df3.iloc[1]
    inCommon = set(Females)&set(Males)
    print("Zad10. Wspolne imiona: ")
    #print(inCommon)
    for i in inCommon:
        counter+=1
    print(counter)
    A = inCommon
    df_final = data[data['name'].isin(A)]
    females = df_final.loc[(df_final['sex'] == 'F')]
    males = df_final.loc[(df_final['sex'] == 'M')]

    top_females = females.groupby(['name']).agg({'number': 'sum'}).reset_index()
    top_males = males.groupby(['name']).agg({'number': 'sum'}).reset_index()
    top_females = top_females.nlargest(1, ['number'])
    top_males = top_males.nlargest(1, ['number'])

    print(top_females)
    print(top_males)

    print("Zad10. Najpopularniejsze damskie - Mary, Najpopularniejsze meskie - James")

#TODO ZAD 12
def Zad12():
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()

    SQL_F = pd.read_sql_query(
        '''select
        PopName,
        Sex,
        Year,
        Age,
        mx,
        qx,
        ax,
        lx,
        dx,
        LLx,
        Tx,
        ex
        from USA_fltper_1x1''', conn)
    SQL_M = pd.read_sql_query(
        '''select
        PopName,
        Sex,
        Year,
        Age,
        mx,
        qx,
        ax,
        lx,
        dx,
        LLx,
        Tx,
        ex
        from USA_mltper_1x1''', conn)

    df_fem = pd.DataFrame(SQL_F, columns=['PopName','Sex','Year','Age','mx','qx','ax','lx','dx','LLx','Tx','ex'])
    df_male = pd.DataFrame(SQL_M, columns=['PopName','Sex','Year','Age','mx','qx','ax','lx','dx','LLx','Tx','ex'])
    frames = [df_fem,df_male]
    df=pd.concat(frames)

    print("Zad12. Dataframe z danymi")
    print(df)

    conn.close()
#TODO Zad 13
def Zad13():
    files=[]
    tab1=[]
    tab2=[]
    series_list=[]
    ALL=[]
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()

    SQL_F = pd.read_sql_query(
        '''select
        PopName,
        Sex,
        Year,
        Age,
        mx,
        qx,
        ax,
        lx,
        dx,
        LLx,
        Tx,
        ex
        from USA_fltper_1x1''', conn)
    SQL_M = pd.read_sql_query(
        '''select
        PopName,
        Sex,
        Year,
        Age,
        mx,
        qx,
        ax,
        lx,
        dx,
        LLx,
        Tx,
        ex
        from USA_mltper_1x1''', conn)

    # for row in c.execute('SELECT * FROM USA_fltper_1x1'):
    # print(row)

    df_fem = pd.DataFrame(SQL_F,
                          columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx', 'LLx', 'Tx', 'ex'])
    df_male = pd.DataFrame(SQL_M,
                           columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx', 'LLx', 'Tx', 'ex'])
    frames = [df_fem, df_male]
    df = pd.concat(frames)
    #print(df)

    for i in range(1959, 2018):
        file = glob('yob'+str(i)+'.txt')
        file.sort()

    for f in file:
        df = pd.read_csv(f, sep=",", header=None, names=["name", "sex", "number"])
        #print(df)
        Sexes = df.groupby('sex')['number'].sum().reset_index()

        tab1.append(Sexes.iloc[0, 1])
        tab2.append(Sexes.iloc[1, 1])


    for f in df_fem:
        pass
    ALL = [x + y for x, y in zip(tab1, tab2)]
    Total_fem = df_fem['dx'].sum()
    Total_m = df_male['dx'].sum()
    Total = Total_m+Total_fem
    #print(ALL)
    print("Zad13. Przyrost naturalny wynosi: ")
    print(ALL-Total)
    conn.close()


Zad123()
Zad4()
Zad5()
Zad6()
Zad7()
Zad8()
Zad10()
Zad12()
Zad13()
