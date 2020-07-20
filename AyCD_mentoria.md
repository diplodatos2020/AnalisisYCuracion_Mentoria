<center>
<h4>Universidad Nacional de Córdoba - Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones</h4>
<h3> Análisis y Curación de Datos </h3>
<h4> Practico de Mentoria</h4>
</center>



```python

```


```python
# https://github.com/diplodatos2020/Introduccion_Mentoria/blob/master/dataset_inf_telec.csv
%load_ext autoreload
%autoreload 2

%matplotlib inline
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(15, 5)})
filename = "https://raw.githubusercontent.com/diplodatos2020/Introduccion_Mentoria/master/dataset_inf_telec_ayc.csv"
BLUE = '#35A7FF'
RED = '#FF5964'
GREEN = '#6BF178'
YELLOW = '#FFE74C'
```


```python
df = pd.read_csv(filename)
```

1. Importacion de los datos
Elija algun PUNTO MEDICION y calcule el rango que existe en la feature FECHA HORA.

Por ejemplo, el PUNTO MEDICION ABA - Abasto Cliente





```python
df_abasto = df[df['PUNTO MEDICION'] == "ABA - Abasto Cliente"]
df_abasto.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID EQUIPO</th>
      <th>PUNTO MEDICION</th>
      <th>CAPACIDAD MAXIMA [GBS]</th>
      <th>FECHA INICIO MEDICION</th>
      <th>FECHA HORA</th>
      <th>FECHA FIN MEDICION</th>
      <th>PASO</th>
      <th>LATENCIA [MS]</th>
      <th>% PACK LOSS</th>
      <th>INBOUND [BITS]</th>
      <th>OUTBOUND [BITS]</th>
      <th>MEDIDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9981</th>
      <td>25</td>
      <td>ABA - Abasto Cliente</td>
      <td>1.0</td>
      <td>2020-06-04 17:00:00.000</td>
      <td>2020-06-16 13:00:00.000</td>
      <td>2020-06-21 19:00:00.000</td>
      <td>7200.0</td>
      <td>0.72051</td>
      <td>0.0</td>
      <td>3.754367e+06</td>
      <td>5.158396e+06</td>
      <td>MB</td>
    </tr>
    <tr>
      <th>46</th>
      <td>25</td>
      <td>ABA - Abasto Cliente</td>
      <td>1.0</td>
      <td>2020-06-04 17:00:00.000</td>
      <td>2020-06-08 15:00:00.000</td>
      <td>2020-06-21 19:00:00.000</td>
      <td>7200.0</td>
      <td>0.61405</td>
      <td>0.0</td>
      <td>8.015315e+06</td>
      <td>1.247106e+07</td>
      <td>MB</td>
    </tr>
    <tr>
      <th>133</th>
      <td>25</td>
      <td>ABA - Abasto Cliente</td>
      <td>1.0</td>
      <td>2020-06-04 17:00:00.000</td>
      <td>2020-06-15 21:00:00.000</td>
      <td>2020-06-21 19:00:00.000</td>
      <td>7200.0</td>
      <td>0.72501</td>
      <td>0.0</td>
      <td>2.911362e+06</td>
      <td>5.187593e+06</td>
      <td>MB</td>
    </tr>
    <tr>
      <th>142</th>
      <td>25</td>
      <td>ABA - Abasto Cliente</td>
      <td>1.0</td>
      <td>2020-06-04 17:00:00.000</td>
      <td>2020-06-16 15:00:00.000</td>
      <td>2020-06-21 19:00:00.000</td>
      <td>7200.0</td>
      <td>0.71063</td>
      <td>0.0</td>
      <td>2.550768e+06</td>
      <td>6.134399e+06</td>
      <td>MB</td>
    </tr>
    <tr>
      <th>9925</th>
      <td>25</td>
      <td>ABA - Abasto Cliente</td>
      <td>1.0</td>
      <td>2020-06-04 17:00:00.000</td>
      <td>2020-06-11 21:00:00.000</td>
      <td>2020-06-21 19:00:00.000</td>
      <td>7200.0</td>
      <td>0.61363</td>
      <td>0.0</td>
      <td>3.712841e+06</td>
      <td>1.020418e+07</td>
      <td>MB</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Descomentando esta linea aparece el error de querer operar con fechas sin el formato adecuado

#df_abasto['FECHA HORA'].max() - df_abasto['FECHA HORA'].min()
```


```python
df.dtypes
```




    ID EQUIPO                   int64
    PUNTO MEDICION             object
    CAPACIDAD MAXIMA [GBS]    float64
    FECHA INICIO MEDICION      object
    FECHA HORA                 object
    FECHA FIN MEDICION         object
    PASO                      float64
    LATENCIA [MS]             float64
    % PACK LOSS               float64
    INBOUND [BITS]            float64
    OUTBOUND [BITS]           float64
    MEDIDA                     object
    dtype: object



Los campos object generalmente son String, entonces parece que no reconoció como fechas en "FECHA_HORA","FECHA_INICIO_MEDICION","FECHA_FIN_MEDICION" :


```python
df = pd.read_csv(filename,parse_dates=["FECHA HORA","FECHA INICIO MEDICION","FECHA FIN MEDICION"])
df.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID EQUIPO</th>
      <th>PUNTO MEDICION</th>
      <th>CAPACIDAD MAXIMA [GBS]</th>
      <th>FECHA INICIO MEDICION</th>
      <th>FECHA HORA</th>
      <th>FECHA FIN MEDICION</th>
      <th>PASO</th>
      <th>LATENCIA [MS]</th>
      <th>% PACK LOSS</th>
      <th>INBOUND [BITS]</th>
      <th>OUTBOUND [BITS]</th>
      <th>MEDIDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19680.000000</td>
      <td>19680</td>
      <td>19680.000000</td>
      <td>19680</td>
      <td>19680</td>
      <td>19680</td>
      <td>18505.0</td>
      <td>18485.000000</td>
      <td>18504.000000</td>
      <td>1.848000e+04</td>
      <td>1.846900e+04</td>
      <td>19680</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>48</td>
      <td>NaN</td>
      <td>1</td>
      <td>205</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>SF - SF Cliente</td>
      <td>NaN</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-18 21:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MB</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>410</td>
      <td>NaN</td>
      <td>19680</td>
      <td>96</td>
      <td>19680</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11890</td>
    </tr>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-04 19:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>last</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.250000</td>
      <td>NaN</td>
      <td>6.211654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>2.816634</td>
      <td>0.203675</td>
      <td>6.503149e+08</td>
      <td>1.336992e+09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.429466</td>
      <td>NaN</td>
      <td>8.264031</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.132946</td>
      <td>0.926476</td>
      <td>2.381072e+09</td>
      <td>3.097376e+09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.027263</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>0.250300</td>
      <td>0.000000</td>
      <td>1.018562e+02</td>
      <td>3.633833e+03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>1.276120</td>
      <td>0.000000</td>
      <td>1.296367e+06</td>
      <td>2.990697e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>2.031490</td>
      <td>0.000000</td>
      <td>1.482903e+07</td>
      <td>8.352007e+07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31.000000</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>3.537790</td>
      <td>0.145930</td>
      <td>2.265187e+08</td>
      <td>9.981229e+08</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>62.000000</td>
      <td>NaN</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7200.0</td>
      <td>27.051760</td>
      <td>41.522270</td>
      <td>2.418785e+10</td>
      <td>2.327810e+10</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    ID EQUIPO                          int64
    PUNTO MEDICION                    object
    CAPACIDAD MAXIMA [GBS]           float64
    FECHA INICIO MEDICION     datetime64[ns]
    FECHA HORA                datetime64[ns]
    FECHA FIN MEDICION        datetime64[ns]
    PASO                             float64
    LATENCIA [MS]                    float64
    % PACK LOSS                      float64
    INBOUND [BITS]                   float64
    OUTBOUND [BITS]                  float64
    MEDIDA                            object
    dtype: object



Ahora podemos ver que las columnas mencionadas son de tipo fecha


```python
df_abasto = df[df['PUNTO MEDICION'] == "ABA - Abasto Cliente"]
```


```python
 df_abasto['FECHA HORA'].max() - df_abasto['FECHA HORA'].min()
```




    Timedelta('17 days 00:00:00')



2. **Etiquetas de variables/columnas: no usar caracteres especiales**
Chequear que no haya caracteres fuera de a-Z, 0-9 y _ en los nombres de columnas del Dataframe.



```python
columns_orig = df.columns
columns_orig
```




    Index(['ID EQUIPO', 'PUNTO MEDICION', 'CAPACIDAD MAXIMA [GBS]',
           'FECHA INICIO MEDICION', 'FECHA HORA', 'FECHA FIN MEDICION', 'PASO',
           'LATENCIA [MS]', '% PACK LOSS', 'INBOUND [BITS]', 'OUTBOUND [BITS]',
           'MEDIDA'],
          dtype='object')




```python
columns_orig.str.match(r'^([\w\d_]+)$')
```




    array([False, False, False, False, False, False,  True, False, False,
           False, False,  True])



Vemos que muchos no cumplen con la condicion de solo incluir letras, numeros y guion bajo.


```python
df.columns = ['ID_EQUIPO', 'PUNTO_MEDICION', 'CAPACIDAD_MAXIMA', 'FECHA_INICIO_MEDICION', 'FECHA_HORA', 'FECHA_FIN_MEDICION', 'PASO', 'LATENCIA', 'PACK_LOSS',
              'INBOUND_BITS', 'OUTBOUND_BITS', 'MEDIDA']
```


```python
df.columns
```




    Index(['ID_EQUIPO', 'PUNTO_MEDICION', 'CAPACIDAD_MAXIMA',
           'FECHA_INICIO_MEDICION', 'FECHA_HORA', 'FECHA_FIN_MEDICION', 'PASO',
           'LATENCIA', 'PACK_LOSS', 'INBOUND_BITS', 'OUTBOUND_BITS', 'MEDIDA'],
          dtype='object')




```python
df.columns.str.match(r'^([\w_\d]+)$')
```




    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True])




3. Agregar nuevas caracteristicas

Agregar al Dataframe dos nuevas columnas INBOUND y OUTBOUND que seran las columnas INBOUND_BITS y OUTBOUND_BITS llevadas a la unidad especificada en la columna MEDIDA.



```python
set(df['MEDIDA'])
```




    {'GB', 'MB'}




```python
df.loc[:,('INBOUND')]=df['INBOUND_BITS'] * 8 * 1024
df.loc[:,('OUTBOUND')]=df['OUTBOUND_BITS'] * 8 * 1024
```

Primero convierto de Bits a Bytes, luego a MBytes, y finalmente multiplico por 1024 los que tengan GB en el campo MEDIDA


```python
df.loc[df['MEDIDA']=='GB', 'INBOUND'] = df.loc[df['MEDIDA']=='GB', 'INBOUND'] * 1024
df.loc[df['MEDIDA']=='GB', 'OUTBOUND'] = df.loc[df['MEDIDA']=='GB', 'OUTBOUND'] * 1024
```

**4. Tratar valores faltantes**
Veamos cuantos valores nulos tenemos:


```python
[df_missing_values_count > 0]
```




    [ID_EQUIPO                False
     PUNTO_MEDICION           False
     CAPACIDAD_MAXIMA         False
     FECHA_INICIO_MEDICION    False
     FECHA_HORA               False
     FECHA_FIN_MEDICION       False
     PASO                      True
     LATENCIA                  True
     PACK_LOSS                 True
     INBOUND_BITS              True
     OUTBOUND_BITS             True
     MEDIDA                   False
     INBOUND                   True
     OUTBOUND                  True
     dtype: bool]




```python
def contar_nan(dfaux):
  df_missing_values_count = dfaux.isna().sum()
  print(df_missing_values_count[df_missing_values_count > 0])
contar_nan(df)
```

    PASO             1175
    LATENCIA         1195
    PACK_LOSS        1176
    INBOUND_BITS     1200
    OUTBOUND_BITS    1211
    INBOUND          1200
    OUTBOUND         1211
    dtype: int64
    


```python
porcentaje = 100 - len(df.dropna(subset=['LATENCIA']))/len(df) * 100
print(f"Porcentaje de filas con valores nulos en el campo LATENCIA frente al total: %{porcentaje:.2}")
```

    Porcentaje de filas con valores nulos en el campo LATENCIA frente al total: %6.1
    

Vemos que todas las columnas con valores faltantes carecen de aproximadamente la misma cantidad de valores, que representan un 6% del total. 
Este porcentaje puede acumularse entre los faltantes de las distintas columnas, entonces evaluamos que no es conveniente eliminar dichas entradas, ya que son considerables en realcion al tamano del dataset.

**Inputacion usando Media y Moda**

A continuacion enumeramos las tres maneras de imputar valores NaN


```python
df_1 = df.copy()
df_1["LATENCIA"].fillna(df_1["LATENCIA"].mean(), inplace = True) # Inputacion con media
contar_nan(df_1)
```

    PASO             1175
    PACK_LOSS        1176
    INBOUND_BITS     1200
    OUTBOUND_BITS    1211
    INBOUND          1200
    OUTBOUND         1211
    dtype: int64
    


```python
df_2 = df.copy()
df_2["LATENCIA"].fillna(df_2["LATENCIA"].mode()[0], inplace = True) # Inputacion con moda
contar_nan(df_2)
```

    PASO             1175
    PACK_LOSS        1176
    INBOUND_BITS     1200
    OUTBOUND_BITS    1211
    INBOUND          1200
    OUTBOUND         1211
    dtype: int64
    


```python
df_3 = df.copy()
df_3["LATENCIA"].fillna(df_3["LATENCIA"].median(), inplace = True) # Inputacion con moda
contar_nan(df_3)
```

    PASO             1175
    PACK_LOSS        1176
    INBOUND_BITS     1200
    OUTBOUND_BITS    1211
    INBOUND          1200
    OUTBOUND         1211
    dtype: int64
    

De las tres, perefimos usar la mediana, ya que no se ve afectada por valores muy extremos.


```python
df_missing_values_count = df.isna().sum()
columnas_corruptas = df_missing_values_count[df_missing_values_count > 0].keys()
columnas_corruptas
```




    Index(['PASO', 'LATENCIA', 'PACK_LOSS', 'INBOUND_BITS', 'OUTBOUND_BITS',
           'INBOUND', 'OUTBOUND'],
          dtype='object')




```python
df_4 = df.copy()
for columna in columnas_corruptas:
  df_4[columna].fillna(df_4[columna].median(), inplace = True) 
contar_nan(df_4)
```

    Series([], dtype: int64)
    


```python
df.size == df_4.size # vemos que el tamano del dataset no se altero
```




    True




5. Codificar variables

    Las variables categóricas deben ser etiquetadas como variables numéricas, no como cadenas.

Codificar la variable PUNTO MEDICION del Dataframe.



```python
from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
label_encoding.fit(df.PUNTO_MEDICION)
label_encoding.classes_
```




    array(['ABA - Abasto Cliente', 'ABA - Temple', 'BAZ - Carlos Paz',
           'BAZ - Yocsina', 'Carlos Paz - Cosquin', 'Carlos Paz - La Falda',
           'EDC - Capitalinas', 'EDC - Coral State', 'EDC - ET Oeste',
           'EDC - MOP', 'EDC - NOR', 'EDC - RDB', 'EDC - Telecomunicacioes',
           'EDC - Transporte', 'JM - Totoral Nueva', 'JM - Totoral Vieja',
           'NOC - 6720HI to BAZ', 'NOC - 6720HI to EDC',
           'NOC - 6720HI to ETC', 'NOC - 6720HI to N20-1',
           'NOC - 6720HI to R4 Silica', 'NOC - 6720HI to RPrivado',
           'NOC - ACHALA - Servicios', 'NOC - ACHALA - Solo Dolores',
           'NOC - Almacenes', 'NOC - ET Sur', 'NOC - Interfabricas',
           'NOC - Pilar', 'NOC - S9306 to SS6720HI', 'NOC - SW Clientes 1',
           'NOC - SW Clientes 2', 'NOC - Switch Servers', 'NOC - UTN',
           'RDB - ET Don Bosco - San Roque', 'RDB - ET La Calera',
           'RDB - Escuela de Capacitacion', 'RDB - GZU', 'RDB - JM',
           'RDB - PEA', 'RDB - RIO', 'SF - Freyre', 'SF - La Francia',
           'SF - Las Varillas', 'SF - SF Adm', 'SF - SF Cliente',
           'Yocsina - Alta Gracia', 'Yocsina - Carlos Paz',
           'Yocsina - Mogote'], dtype=object)




```python
df.sample(10,random_state = 99)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_EQUIPO</th>
      <th>PUNTO_MEDICION</th>
      <th>CAPACIDAD_MAXIMA</th>
      <th>FECHA_INICIO_MEDICION</th>
      <th>FECHA_HORA</th>
      <th>FECHA_FIN_MEDICION</th>
      <th>PASO</th>
      <th>LATENCIA</th>
      <th>PACK_LOSS</th>
      <th>INBOUND_BITS</th>
      <th>OUTBOUND_BITS</th>
      <th>MEDIDA</th>
      <th>INBOUND</th>
      <th>OUTBOUND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14106</th>
      <td>62</td>
      <td>NOC - 6720HI to R4 Silica</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-18 15:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>0.57050</td>
      <td>0.00000</td>
      <td>2.599131e+07</td>
      <td>3.680639e+08</td>
      <td>MB</td>
      <td>2.129208e+11</td>
      <td>3.015180e+12</td>
    </tr>
    <tr>
      <th>3318</th>
      <td>62</td>
      <td>NOC - 6720HI to BAZ</td>
      <td>20.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-07 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>1.63787</td>
      <td>0.00000</td>
      <td>1.335610e+09</td>
      <td>1.509467e+10</td>
      <td>GB</td>
      <td>1.120391e+16</td>
      <td>1.266233e+17</td>
    </tr>
    <tr>
      <th>11527</th>
      <td>24</td>
      <td>EDC - ET Oeste</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 17:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.55728</td>
      <td>0.20889</td>
      <td>2.122304e+07</td>
      <td>4.731276e+06</td>
      <td>MB</td>
      <td>1.738592e+11</td>
      <td>3.875862e+10</td>
    </tr>
    <tr>
      <th>859</th>
      <td>30</td>
      <td>Carlos Paz - Cosquin</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 01:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>3.08726</td>
      <td>0.58694</td>
      <td>6.261134e+07</td>
      <td>7.453256e+08</td>
      <td>MB</td>
      <td>5.129121e+11</td>
      <td>6.105707e+12</td>
    </tr>
    <tr>
      <th>4574</th>
      <td>41</td>
      <td>NOC - ACHALA - Servicios</td>
      <td>0.027263</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-10 03:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.84778</td>
      <td>0.00000</td>
      <td>4.191129e+05</td>
      <td>3.981020e+06</td>
      <td>MB</td>
      <td>3.433373e+09</td>
      <td>3.261251e+10</td>
    </tr>
    <tr>
      <th>13077</th>
      <td>28</td>
      <td>JM - Totoral Vieja</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-18 07:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.81624</td>
      <td>0.00000</td>
      <td>1.117076e+08</td>
      <td>4.018887e+08</td>
      <td>GB</td>
      <td>9.370715e+14</td>
      <td>3.371287e+15</td>
    </tr>
    <tr>
      <th>12956</th>
      <td>28</td>
      <td>JM - Totoral Vieja</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 05:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.79798</td>
      <td>0.47717</td>
      <td>1.157807e+08</td>
      <td>4.827438e+08</td>
      <td>GB</td>
      <td>9.712385e+14</td>
      <td>4.049548e+15</td>
    </tr>
    <tr>
      <th>7635</th>
      <td>31</td>
      <td>RDB - JM</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>5.09392</td>
      <td>0.14549</td>
      <td>6.922368e+08</td>
      <td>7.096758e+09</td>
      <td>GB</td>
      <td>5.806904e+15</td>
      <td>5.953192e+16</td>
    </tr>
    <tr>
      <th>8536</th>
      <td>11</td>
      <td>SF - La Francia</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-15 17:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>6.39851</td>
      <td>0.00000</td>
      <td>2.546614e+09</td>
      <td>1.852100e+08</td>
      <td>GB</td>
      <td>2.136254e+16</td>
      <td>1.553654e+15</td>
    </tr>
    <tr>
      <th>16257</th>
      <td>4</td>
      <td>NOC - Switch Servers</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-09 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>1.94908</td>
      <td>0.00000</td>
      <td>6.878710e+04</td>
      <td>2.984526e+05</td>
      <td>MB</td>
      <td>5.635039e+08</td>
      <td>2.444924e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.PUNTO_MEDICION = label_encoding.transform(df.PUNTO_MEDICION)
```


```python
df.sample(10,random_state = 99)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_EQUIPO</th>
      <th>PUNTO_MEDICION</th>
      <th>CAPACIDAD_MAXIMA</th>
      <th>FECHA_INICIO_MEDICION</th>
      <th>FECHA_HORA</th>
      <th>FECHA_FIN_MEDICION</th>
      <th>PASO</th>
      <th>LATENCIA</th>
      <th>PACK_LOSS</th>
      <th>INBOUND_BITS</th>
      <th>OUTBOUND_BITS</th>
      <th>MEDIDA</th>
      <th>INBOUND</th>
      <th>OUTBOUND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14106</th>
      <td>62</td>
      <td>20</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-18 15:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>0.57050</td>
      <td>0.00000</td>
      <td>2.599131e+07</td>
      <td>3.680639e+08</td>
      <td>MB</td>
      <td>2.129208e+11</td>
      <td>3.015180e+12</td>
    </tr>
    <tr>
      <th>3318</th>
      <td>62</td>
      <td>16</td>
      <td>20.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-07 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>1.63787</td>
      <td>0.00000</td>
      <td>1.335610e+09</td>
      <td>1.509467e+10</td>
      <td>GB</td>
      <td>1.120391e+16</td>
      <td>1.266233e+17</td>
    </tr>
    <tr>
      <th>11527</th>
      <td>24</td>
      <td>8</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 17:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.55728</td>
      <td>0.20889</td>
      <td>2.122304e+07</td>
      <td>4.731276e+06</td>
      <td>MB</td>
      <td>1.738592e+11</td>
      <td>3.875862e+10</td>
    </tr>
    <tr>
      <th>859</th>
      <td>30</td>
      <td>4</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 01:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>3.08726</td>
      <td>0.58694</td>
      <td>6.261134e+07</td>
      <td>7.453256e+08</td>
      <td>MB</td>
      <td>5.129121e+11</td>
      <td>6.105707e+12</td>
    </tr>
    <tr>
      <th>4574</th>
      <td>41</td>
      <td>22</td>
      <td>0.027263</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-10 03:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.84778</td>
      <td>0.00000</td>
      <td>4.191129e+05</td>
      <td>3.981020e+06</td>
      <td>MB</td>
      <td>3.433373e+09</td>
      <td>3.261251e+10</td>
    </tr>
    <tr>
      <th>13077</th>
      <td>28</td>
      <td>15</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-18 07:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.81624</td>
      <td>0.00000</td>
      <td>1.117076e+08</td>
      <td>4.018887e+08</td>
      <td>GB</td>
      <td>9.370715e+14</td>
      <td>3.371287e+15</td>
    </tr>
    <tr>
      <th>12956</th>
      <td>28</td>
      <td>15</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 05:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>2.79798</td>
      <td>0.47717</td>
      <td>1.157807e+08</td>
      <td>4.827438e+08</td>
      <td>GB</td>
      <td>9.712385e+14</td>
      <td>4.049548e+15</td>
    </tr>
    <tr>
      <th>7635</th>
      <td>31</td>
      <td>37</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-08 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>5.09392</td>
      <td>0.14549</td>
      <td>6.922368e+08</td>
      <td>7.096758e+09</td>
      <td>GB</td>
      <td>5.806904e+15</td>
      <td>5.953192e+16</td>
    </tr>
    <tr>
      <th>8536</th>
      <td>11</td>
      <td>41</td>
      <td>10.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-15 17:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>6.39851</td>
      <td>0.00000</td>
      <td>2.546614e+09</td>
      <td>1.852100e+08</td>
      <td>GB</td>
      <td>2.136254e+16</td>
      <td>1.553654e+15</td>
    </tr>
    <tr>
      <th>16257</th>
      <td>4</td>
      <td>31</td>
      <td>1.000000</td>
      <td>2020-06-04 17:00:00</td>
      <td>2020-06-09 23:00:00</td>
      <td>2020-06-21 19:00:00</td>
      <td>7200.0</td>
      <td>1.94908</td>
      <td>0.00000</td>
      <td>6.878710e+04</td>
      <td>2.984526e+05</td>
      <td>MB</td>
      <td>5.635039e+08</td>
      <td>2.444924e+09</td>
    </tr>
  </tbody>
</table>
</div>



Vemos por ejemplo en la fila 13077 y 12956, que correspondian a JM - Totoral Vieja, recibieron el mismo encoding, por lo que verificamos una codificacion exitosa.
