import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



df = pd.read_excel("input/PMO.xlsx")
df = df.drop(columns = ['Caso tenha optado pela opção "Parcialmente", informe o percentual de servidores que não são da organização:','Caso tenha optado pela opção "Sim", informe o modelo.','Dentro da sua organização, quais são as principais dificuldades na implantação de práticas de gestão de projetos, programas e/ou portfólio?','Dentro da sua organização, quais são as principais oportunidades na implantação de práticas de gestão de projetos, programas e/ou portfólio?','São funções do PMO da sua organização:'])

# df["nota"] = df["O desempenho em prazos dos projetos desenvolvidos pela sua organização é:"]




for coluna in list(df):


    unicos = df[coluna].unique().tolist()
    if all(elem in unicos for elem in ['Mais de 5000 profissionais', 'Entre 21 e 50 profisisonais',
 'Até 20 profissionais' ,'Entre 251 e 500 profisisonais',
 'Entre 51 e 100 profisisonais' ,'Entre 501 e 1000 profisisonais',
 'Entre 1001 e 5000 profisisonais' ,'Entre 101 e 250 profisisonais']):
        unicos = ['Até 20 profissionais',
                  'Entre 21 e 50 profisisonais',
                  'Entre 51 e 100 profisisonais',
                  'Mais de 5000 profissionais',
                  'Entre 101 e 250 profisisonais',
                  'Entre 251 e 500 profisisonais',
                  'Entre 501 e 1000 profisisonais',
                  'Entre 1001 e 5000 profisisonais' ]

    elif all(elem in unicos for elem in ['Menos de 1 ano',
                           'Minha organização não possui PMO',
                           'Mais de 5 anos',
                            'Entre 1 e 2 anos',
                           'Entre 3 e 5 anos']):
        unicos = ['Minha organização não possui PMO',
                  'Menos de 1 ano',
                  'Entre 1 e 2 anos',
                  'Entre 3 e 5 anos',
                    'Mais de 5 anos']
    elif all(elem in unicos for elem in [1, 0, '5 ou mais', 2, 3, 4]):
        unicos = [0,1, 2, 3, 4,'5 ou mais']
        unicos =  [str(x) for x in unicos]



    elif all(elem in unicos for elem in ['Estratégico', 'Estratégico, Tático, Operacional', 'Operacional', 'Estratégico, Tático', 'Tático', 'Estratégico, Operacional', 'Tático, Operacional']):
        unicos = ['Operacional', 'Tático','Tático, Operacional','Estratégico',  'Estratégico, Operacional','Estratégico, Tático','Estratégico, Tático, Operacional']

    elif all(elem in unicos for elem in ['Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização, Todas as anteriores', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter ferramentas', 'Todas as anteriores', 'Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Desenvolver/manter os métodos', 'Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Gerenciar os projetos da organização', 'Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização, Todas as anteriores', 'Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização', 'Capacitação', 'Capacitação, Gerenciar os projetos da organização', 'Sensibilização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Gerenciar os projetos da organização', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Gerenciar os projetos da organização', 'Sensibilização, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização']):
        unicos = ['Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização, Todas as anteriores', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter ferramentas',
                  'Todas as anteriores',
                  'Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter os métodos',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Desenvolver/manter os métodos', 'Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter ferramentas', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização, Desenvolver/manter os métodos',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Gerenciar os projetos da organização',
                  'Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Desenvolver/manter ferramentas',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização, Todas as anteriores',
                  'Desenvolver/manter os métodos, Desenvolver/manter ferramentas, Gerenciar os projetos da organização',
                  'Capacitação',
                  'Capacitação, Gerenciar os projetos da organização',
                  'Sensibilização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos, Gerenciar os projetos da organização',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos, Gerenciar os projetos da organização',
                  'Sensibilização, Desenvolver/manter os métodos', 'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Sensibilização, Desenvolver/manter os métodos',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Desenvolver/manter os métodos',
                  'Apoio Técnico e Metodológico aos gerentes de projetos, programas e portfólio, Capacitação, Sensibilização']

    elif all(elem in unicos for elem in ['Baixa, é um departamento burocrático', 'Alta, o PMO agrega valor para a organização', 'Média, é uma instância de controle apenas', 'Não sei informar', 'Alta, o PMO agrega valor para a organização, Média, é uma instância de controle apenas', 'Média, é uma instância de controle apenas, Baixa, é um departamento burocrático', 'Alta, o PMO agrega valor para a organização, Média, é uma instância de controle apenas, Baixa, é um departamento burocrático']):


        unicos = ['Não sei informar',
                  'Baixa, é um departamento burocrático',
                  'Média, é uma instância de controle apenas, Baixa, é um departamento burocrático',
                  'Média, é uma instância de controle apenas',
                  'Alta, o PMO agrega valor para a organização, Média, é uma instância de controle apenas, Baixa, é um departamento burocrático',
                  'Alta, o PMO agrega valor para a organização, Média, é uma instância de controle apenas',
                  'Alta, o PMO agrega valor para a organização',
                  ]
    elif all(elem in unicos for elem in ['Entre 2 e 4 pessoas', '1 pessoa', 'Mais de 15 pessoas', 'Entre 5 e 10 pessoas', 'Entre 11 e 15 pessoas']):

        unicos = ['1 pessoa','Entre 2 e 4 pessoas', 'Entre 5 e 10 pessoas', 'Entre 11 e 15 pessoas', 'Mais de 15 pessoas', ]

    elif all(elem in unicos for elem in ['Sim', 'Parcialmente', 'Não']):

        unicos = [ 'Não','Parcialmente','Sim', ]

    elif all(elem in unicos for elem in ["nan", '0% a 20%', '41% a 60%', '61% a 80%', '21% a 40%']):

        unicos = ["nan", '0% a 20%',  '21% a 40%','41% a 60%', '61% a 80%']
        print("HELLO")

    elif all(elem in unicos for elem in ['Portfólio, Projetos',
                    'Portfólio, Programas, Projetos',
                    'Projetos', 'Programas, Projetos',
                    'Nenhum',
                    'Portfólio',
                    'Programas',
                    'Portfólio, Programas']):

        unicos = ['Nenhum',
                  'Projetos',
                  'Programas',
                  'Programas, Projetos',
                  'Portfólio',
                  'Portfólio, Projetos',
                  'Portfólio, Programas',
                    'Portfólio, Programas, Projetos',
                    ]
    elif all(elem in unicos for elem in ['Entre 3 e 5 anos', 'Não sei informar', 'Mais de 5 anos', 'Entre 1 e 2 anos', 'Menos de 1 ano']):

        unicos = ['Não sei informar',
                    'Menos de 1 ano',
                    'Entre 1 e 2 anos',
                    'Entre 3 e 5 anos',
                    'Mais de 5 anos',
                    ]
    elif all(elem in unicos for elem in
             ['Raramente', 'Nunca', 'Com alta Frequência', 'Com média frequência', "Não sei informar"]):

        unicos = ["Não sei informar", 'Nunca', 'Raramente', 'Com média frequência', 'Com alta Frequência']

    elif all(elem in unicos for elem in ['Não', 'Sim', 'Não sei informar', 'Somente para alguns']):

        unicos = ['Não sei informar','Não', 'Somente para alguns','Sim']

    elif all(elem in unicos for elem in ['Sim', 'Não', 'Não sei informar']):

        unicos = [ 'Não', 'Não sei informar','Sim']

    elif all(elem in unicos for elem in ['Raramente', 'Não sei informar', 'Frequentemente', 'Sempre', 'Nunca']):

        unicos = ['Não sei informar', 'Nunca','Raramente',  'Frequentemente', 'Sempre']


    elif all(elem in unicos for elem in ['Sim, com frequência mediana', 'Sim, com pouca frequência', 'Sim, com muita frequência', 'Não']):

        unicos = ['Não','Sim, com pouca frequência','Sim, com frequência mediana',  'Sim, com muita frequência', ]

    elif all(elem in unicos for elem in ['Raramente', 'Nunca', 'Com alta Frequência', 'Com média frequência']):

        unicos = ['Nunca','Raramente','Com média frequência','Com alta Frequência' ]



    elif all(elem in unicos for elem in ['Raramente', 'Nunca','Frequentemente','Não sei informar','Sempre']):

        unicos = ['Nunca','Raramente','Não sei informar','Frequentemente','Sempre']



    elif all(elem in unicos for elem in ['de 10 a 50 mil Reais', 'Não sei informar', 'Não há orçamento destinado a isso', 'Mais de 50 mil Reais',
     'de 5 a 10 mil Reais', 'até 5 mil Reais']):

        unicos = ['Não sei informar',
                  'Não há orçamento destinado a isso',
                  'até 5 mil Reais',
                  'de 5 a 10 mil Reais',
                  'de 10 a 50 mil Reais',
                  'Mais de 50 mil Reais',
                  ]
    elif all(elem in unicos for elem in ['Os projetos destoam PARCIALMENTE dos prazos previstos', 'Os projetos atendem aos prazos previstos', 'Os projetos atendem PARCIALMENTE aos prazos previstos', 'Os projetos destoam TOTALMENTE dos prazos previstos', 'Os projetos atendem TOTALMENTE aos prazos previstos']):

        unicos = ['Os projetos destoam TOTALMENTE dos prazos previstos',
                  'Os projetos atendem PARCIALMENTE aos prazos previstos',
                  'Os projetos destoam PARCIALMENTE dos prazos previstos',
                  'Os projetos atendem aos prazos previstos',
                  'Os projetos atendem TOTALMENTE aos prazos previstos']

    elif all(elem in unicos for elem in ['Os projetos atendem PARCIALMENTE ao escopo previsto', 'Os projetos atendem ao escopo previsto', 'Os projetos atendem TOTALMENTE ao escopo previsto', 'Os projetos destoam TOTALMENTE do escopo previsto', 'Os projetos destoam PARCIALMENTE do escopo previsto']):

        unicos = ['Os projetos destoam TOTALMENTE do escopo previsto',
                  'Os projetos atendem PARCIALMENTE ao escopo previsto',
                  'Os projetos destoam PARCIALMENTE do escopo previsto',
                  'Os projetos atendem ao escopo previsto',
                  'Os projetos atendem TOTALMENTE ao escopo previsto',
              ]
    elif all(elem in unicos for elem in ['Os projetos destoam PARCIALMENTE dos custos previstos', 'Os projetos atendem aos custos previstos', 'Os projetos destoam TOTALMENTE dos custos previstos', 'Os projetos atendem PARCIALMENTE aos custos previstos', 'Os projetos atendem TOTALMENTE aos custos previstos']):

        unicos = ['Os projetos destoam TOTALMENTE dos custos previstos',
                  'Os projetos atendem PARCIALMENTE aos custos previstos',
                  'Os projetos destoam PARCIALMENTE dos custos previstos',
                  'Os projetos atendem aos custos previstos',
                  'Os projetos atendem TOTALMENTE aos custos previstos']





    # print(unicos)
    # print("\n")
    # print(unicos)

    def mapa(x,i, p):
        if x == p:
            return i
        elif isinstance(x,str) or isinstance(x,int):
            return x
        else:
            return 0

    for i, p in enumerate(unicos):
        df[coluna] = df[coluna].apply(lambda x: mapa(x, i, p))
    # print(df[coluna][df[coluna].isin(range(15))==False])
    # print(unicos)
    # print('\n')
    df[coluna] = df[coluna].apply(lambda x: int(x))
    # print(df)

df["nota"] = df['O desempenho em prazos dos projetos desenvolvidos pela sua organização é:'] + df['O desempenho em escopo dos projetos desenvolvidos pela sua organização é:'] + df['O desempenho em custos dos projetos desenvolvidos pela sua organização é:']
df = df.drop(columns = ['O desempenho em prazos dos projetos desenvolvidos pela sua organização é:','O desempenho em escopo dos projetos desenvolvidos pela sua organização é:','O desempenho em custos dos projetos desenvolvidos pela sua organização é:'])




correlated_features = set()
correlation_matrix = df.drop('nota', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

print(correlated_features)




X = df.drop('nota', axis=1)
target = df['nota']

# X, X_test, target, y_test = train_test_split(X, target, test_size=.2, random_state=42)

# print(y_test)

rfc = RandomForestClassifier(random_state=101, n_estimators= 100)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)

# y_pred = rfecv.predict(X_test)

# cm=confusion_matrix(y_test,y_pred)

# for x, y in zip(y_pred,y_test):
#     print(x)
#     print(y)
#     print("\n")
#
# plt.plot(y_pred, y_test, 'ro')
# plt.axis([0, 13, 0, 13])
# plt.show()
#
# print(cm)
# print(classification_report(y_test, y_pred))

print('Optimal number of features: {}'.format(rfecv.n_features_))

print(df.shape)

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()

for i in np.where(rfecv.support_ == False)[0]:
    print(list(df)[i])

X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)





def correct(x):
    if len(x) > 100:
        return x[0:90] + "\n" + x[90:]
    else:
        return x


dset = pd.DataFrame()
dset['attr'] = X.columns
dset['attr'] = dset['attr'].apply(lambda x: correct(x) )
print(dset['attr'])
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)


plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()