import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline


data = pd.read_csv('/Users/adrianinfantesromero/Desktop/AIR/Work/GitHub/Practicum/Vehicles/vehicles_us.csv')

# Data cleaning

data.head()
data.info()


# la columna date_posted nos indica la fecha en la que se publicó el anuncio y la columna days_listed nos indica el número de días que el anuncio estuvo publicado.
#por eso vamos a crear una nueva columna que nos indique la fecha en la que se vendió el vehículo

# la nueva columan que se llama date_sold se calcula sumando a la columna date_posted el número de días que estuvo publicado el anuncio

data['date_sold'] = pd.to_datetime(data['date_posted']) + pd.to_timedelta(data['days_listed'], unit='d')

# quiero crear una nueva columna que me indique el año en el que se vendió el vehículo en base a la columna date_sold

data['year_sold'] = data['date_sold'].dt.year

# quiero crear una nueva columna que me indique el mes en el que se vendió el vehículo en base a la columna date_sold

data['month_sold'] = data['date_sold'].dt.month

# vamos graficar el número de vehículos vendidos por año y ponerle un título al histograma que sea "Vehicles sold by year"

data['year_sold'].hist(bins=10, figsize=(10,5))
plt.title('Vehicles sold by year')

# vamos graficar el número de vehículos vendidos por mes en el año 2018 y 2019 en un histograma

data.query('year_sold == 2018')['month_sold'].hist(bins=12, figsize=(10,5))
data.query('year_sold == 2019')['month_sold'].hist(bins=12, figsize=(10,5))

#quiero sustituir los valores de la columna model por la primera palabra de la columna model de cada fila

data['model'] = data['model'].str.split().str[0]


#vamos a hacer un histograma en forma de queso para ver la distribución de los vehículos vendidos en 2018 por color

data.query('year_sold == 2018')['paint_color'].hist(bins=10, figsize=(10,5))
data.query('year_sold == 2019')['paint_color'].hist(bins=10, figsize=(10,5))


#cuales son los mejores metodos de visualización para ver la relación entre modelos de coche por años

data.pivot_table(index='model', columns='year_sold', values='price', aggfunc='count').plot(kind='bar', figsize=(10,5))

#tambien quiero ver el color de los vehiculos vendidos en la pivot table

data.pivot_table(index='model', columns='year_sold', values='paint_color', aggfunc='count').plot(kind='bar', figsize=(10,5))# values esta colocado en paint_color para que me muestre el color de los vehiculos vendidos


#vamos a hacer un histograma para ver el total de ventas por la suma de price en 2018 y 2019

data.pivot_table(index='model', columns='year_sold', values='price', aggfunc='sum').plot(kind='bar', figsize=(10,5))


# total de ventas en 2018

data.query('year_sold == 2018')['price'].sum()
data.query('year_sold == 2019')['price'].sum()

# cuanto % de ventas que se vendio mas en 2018 que en 2019 quiero verlo en porcentaje

data.query('year_sold == 2018')['price'].sum() / data.query('year_sold == 2019')['price'].sum() * 100


#vamos a elimnar la columna de odometer porque no nos aporta nada

data = data.drop(['odometer'], axis=1)

#todos los valores NaN de la columna is_4wd los vamos a sustituir por 0.0

data['is_4wd'] = data['is_4wd'].fillna(0.0)


#- Sabemos que los SUV se venden un 7% mas, los convertibles un 0.2%, 1,3% los coupe, los hatchback un 0.6%, los mini-van 0.7%, los offroad 0.2%, los pickup 4.0%, Sedan un 7.0%, truck 7.6%, van 0.3% y por ultimo wagon 0.8%
#quiero sustituir esos valores con los nombres por los float.

data['type'] = data['type'].replace('SUV', 0.07)
data['type'] = data['type'].replace('convertible', 0.002)
data['type'] = data['type'].replace('coupe', 0.013)
data['type'] = data['type'].replace('hatchback', 0.006)
data['type'] = data['type'].replace('mini-van', 0.007)
data['type'] = data['type'].replace('offroad', 0.002)
data['type'] = data['type'].replace('pickup', 0.04)
data['type'] = data['type'].replace('sedan', 0.07)
data['type'] = data['type'].replace('truck', 0.076)
data['type'] = data['type'].replace('van', 0.003)
data['type'] = data['type'].replace('wagon', 0.008)

#- Si el vehiculo es automatic podemos ver que la decision aumenta en 27%
#- Mientras que si es manual es un 3%
#quiero sustituir esos valores con los nombres por los float.

data['type'] = data['type'].replace('auto', 0.27)
data['type'] = data['type'].replace('manual', 0.3)


#Ahora vamos con la decision de fuel
#- Vemos que si el vehiculo es gas es un 27%
#- Mientras que si es diesel es un 3%
#- Por ultimo si es hybrid es un 1%

data['fuel'] = data['fuel'].replace('gas', 0.27)
data['fuel'] = data['fuel'].replace('diesel', 0.03)
data['fuel'] = data['fuel'].replace('hybrid', 0.01)


#Hagamos los mismo con la columna cylinders.
#- si es 4.0 es un 8%
#- 6.0 es un 9%
#- 8.0 es un 10%

data['cylinders'] = data['cylinders'].replace(4.0, 0.08)
data['cylinders'] = data['cylinders'].replace(6.0, 0.09)
data['cylinders'] = data['cylinders'].replace(8.0, 0.1)

#Vamos con la columna condition
#- si la condition es excellent es un 15%
#- fair es 1%
#- good es 11%
#- like new es 3%

data['condition'] = data['condition'].replace('excellent', 0.15)
data['condition'] = data['condition'].replace('fair', 0.01)
data['condition'] = data['condition'].replace('good', 0.11)
data['condition'] = data['condition'].replace('like new', 0.03)


#El color(paint color) afecta tambien:
#- black = 6%
#- white = 5,7%
#- gray = 2.9%
#- green = 0.8%
#- blue = 2.5%
#- custom = 0.6%
#- brown = 0.6%
#- red = 2.6%
#- orange = 0.2%
#- yellow = 0.3%
#- purple = 0.1%
#- silver = 3.6%

data['paint_color'] = data['paint_color'].replace('black', 0.06)
data['paint_color'] = data['paint_color'].replace('white', 0.057)
data['paint_color'] = data['paint_color'].replace('gray', 0.029)
data['paint_color'] = data['paint_color'].replace('green', 0.008)
data['paint_color'] = data['paint_color'].replace('blue', 0.025)
data['paint_color'] = data['paint_color'].replace('custom', 0.006)
data['paint_color'] = data['paint_color'].replace('brown', 0.006)
data['paint_color'] = data['paint_color'].replace('red', 0.026)
data['paint_color'] = data['paint_color'].replace('orange', 0.002)
data['paint_color'] = data['paint_color'].replace('yellow', 0.003)
data['paint_color'] = data['paint_color'].replace('purple', 0.001)
data['paint_color'] = data['paint_color'].replace('silver', 0.036)

Ahora vamos con el modelo

#acura 0.1
#bmw 0.2
#buick 0.3
#cadillac 0.4
#chevrolet 6.2
#chrysler 0.4
#dodge 0.6
#ford 7.5
#gmc 1.3
#honda 2.1
#hyundai 0.6
#jeep 1.9
#kia 0.3
#mercedes-benz 0
#nissan 1.8
#ram 1.9
#subaru 0.6
#toyota 3.1
#wolkswagen 0.4

data['model'] = data['model'].replace('acura', 0.1)
data['model'] = data['model'].replace('bmw', 0.2)
data['model'] = data['model'].replace('buick', 0.3)
data['model'] = data['model'].replace('cadillac', 0.4)
data['model'] = data['model'].replace('chevrolet', 0.62)
data['model'] = data['model'].replace('chrysler', 0.4)
data['model'] = data['model'].replace('dodge', 0.6)
data['model'] = data['model'].replace('ford', 0.75)
data['model'] = data['model'].replace('gmc', 0.13)
data['model'] = data['model'].replace('honda', 0.21)
data['model'] = data['model'].replace('hyundai', 0.6)
data['model'] = data['model'].replace('jeep', 0.19)
data['model'] = data['model'].replace('kia', 0.3)
data['model'] = data['model'].replace('mercedes-benz', 0)
data['model'] = data['model'].replace('nissan', 0.18)
data['model'] = data['model'].replace('ram', 0.19)
data['model'] = data['model'].replace('subaru', 0.6)
data['model'] = data['model'].replace('toyota', 0.31)
data['model'] = data['model'].replace('wolkswagen', 0.4)



#podemos hacer un modelo de entrenamiento y test para ver que tal funciona el modelo con los datos que tenemos en el dataset

X_train, X_test, y_train, y_test = train_test_split(data[['year', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'paint_color', 'type']], data['price'], test_size=0.2, random_state=0) # esto es para dividir el dataset en 2 partes, una para entrenar y otra para testear
# la libreria sklearn tiene un metodo para dividir el dataset en 2 partes, train y test, para entrenar y testear el modelo
#ahora vamos a entrenar el modelo

regressor = LinearRegression()# creamos un objeto de la clase LinearRegression de la libreria sklearn para entrenar el modelo
# pero no reconoce linear_regression porque no lo hemos importado de la libreria sklearn asi que lo importamos arriba de todo junto con las demas librerias

#que alternativas tenemos para LinearRegression?
#- Ridge. esta es la libreria para regularizar el modelo y evitar el overfitting
#- Lasso
#- ElasticNet
#- SGDRegressor
#- HuberRegressor
regressor.fit(X_train, y_train)

#ahora vamos a testear el modelo

y_pred = regressor.predict(X_test)

print(regressor.coef_)
print(regressor.intercept_)
print(regressor.predict(X_test))

#ahora vamos a ver que tal funciona el modelo

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
