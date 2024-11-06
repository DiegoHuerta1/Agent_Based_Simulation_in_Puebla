# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:18:35 2024

@author: diego
"""


import numpy as np
from scipy.spatial.distance import pdist
import networkx as nx
import random
import colorsys
import igraph


from .Agentes import Coche



'''
Funciones auxiliares para el modelo
'''



# lista de proporciones (proporcionada por la interpolacion)
lista_proporcion = np.array([5.947231949990006e-05,0.00019793690452087675,0.00022786987399239686,0.0008187642253172161,0.005882248958286144,0.02413499085765168,0.0659384313349131,0.10040138805960429,0.07069111773393716,0.05069748649652717,0.05527109394071533,0.05222455101798218,0.07655801553394084,0.083384718562456,0.07377294327220996,0.05381158882913746,0.0474688409059631,0.0472548729550307,0.06755114386883684,0.04261318723897216,0.03696081203547986,0.023260422130372913,0.01455799721274494,0.006260105731907801])


def get_total_steps_and_lambda_function(HORA_INICIAL = 0, HORA_FINAL = 24,
                                        NUM_AGENTES_TOTALES = 100,
                                        STEPS_PER_HORA = 100):
  '''
  Funcion usada para tener parametros (steps y funcion lmabda)
  que tengan sentido en la simulacion con respecto al tiempo

  Se pasan los parametros de tiempo deseados
  Se devuelven los steps totales y la funcion lambda correspondiente
  para especificar en el modelo, y que asi la simulacion tenga sentido
  en cuestion de tiempo
  '''
  # calcular en cuantos steps se completa ese rango de horas
  steps_totales = (HORA_FINAL - HORA_INICIAL) * STEPS_PER_HORA

  # solo considerar de las horas de interes y normalizar
  proporciones_interes = lista_proporcion[HORA_INICIAL:HORA_FINAL]/lista_proporcion[HORA_INICIAL:HORA_FINAL].sum()
  # hacer un diccionario que toma la hora y devuelve al proporcion
  dict_proporcion = {i+HORA_INICIAL:proporciones_interes[i] for i in range(len(proporciones_interes))}
  # el de la ultima hora (solo es un step)
  dict_proporcion[HORA_FINAL] = 0

  # hacer una funcion que segun el step en el que estes te de la hora de la simulacion
  def get_hora_simulacion(step_simulacion):
    return np.floor(HORA_INICIAL + step_simulacion / STEPS_PER_HORA)

  # hacer la funcion que toma el step y da el lambda
  def get_lambda_segun_step(step_simulacion):
    # tomar la hora
    hora = get_hora_simulacion(step_simulacion)
    # ver cuantos agentes se quieren hacer en promedio en esa hora
    agentes_totales_hora = dict_proporcion[hora]*NUM_AGENTES_TOTALES
    # dividir esa suma en todos los steps de esa hora
    # recordar que el valor esperado de una poison es su parametro
    # entonces para el parametro ponemos lo que esperamos
    return agentes_totales_hora/STEPS_PER_HORA

  # devolver lo que se necesita
  return steps_totales, get_lambda_segun_step




def precomputar_distancias_(G, pos):
    '''
    Calcular las distancias euclidianas entre pares de nodos en un grafo de NetworkX.
    '''

    # Crear una lista de posiciones en un array ordenado por los índices de los nodos
    posiciones = np.array([pos[u] for u in G.nodes()])

    # Calcular las distancias euclidianas entre todos los pares de nodos usando pdist y squareform
    distancias_pares = pdist(posiciones, metric='euclidean')

    return distancias_pares



def get_color_aleatorio():
  '''
  Funcion auxiliar para visalizaciones
  Toma un color aleatorio
  '''

  # Genera valores aleatorios para los componentes HSV
  h = random.random()  # Tono: valor entre 0 y 1
  s = random.uniform(0.7, 1.0)  # Saturación alta (entre 0.7 y 1)
  v = random.uniform(0.7, 1.0)  # Brillo alto (entre 0.7 y 1)

  # Convierte el valor HSV a RGB
  r, g, b = colorsys.hsv_to_rgb(h, s, v)

  # Convierte el valor RGB a formato hexadecimal
  return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))




def networkx_2_igraph(G_nx):
  '''
  Funcion auxiliar para convertir un grafo de networkx a igraph
  '''

  # hacer un mapeo de los nodos de G a sus indices
  nodes_2_idx = {v: idx_v for idx_v, v in enumerate(G_nx.nodes())}
  # hacer un mapeo de los indices s los nodos de G
  idx_2_nodes = {idx_v: v for idx_v, v in enumerate(G_nx.nodes())}

  # escribir las aristas usando los indices de los nodos
  aristas_idx = [(nodes_2_idx[u], nodes_2_idx[v]) for u, v in G_nx.edges()]

  # crear un grafo de igraph con estas aristas
  G_igraph = igraph.Graph(aristas_idx, directed = True)

  # devovler
  return {
      "G_igraph": G_igraph,
      "nodes_2_idx": nodes_2_idx,
      "idx_2_nodes": idx_2_nodes
  }




def generar_cuadricula_dirigida(n, d):
    # Crear un grafo dirigido vacío
    G = nx.DiGraph()

    # Añadir nodos y aristas
    for i in range(n):
        for j in range(n):
            # Añadir nodo (i, j)
            G.add_node((i, j))

            # Conexión con el vecino de la derecha (bidireccional)
            if j < n - 1:
              # la direccion depende de la paridad de i respecto a d
              if i%(2*d) == 0:
                G.add_edge((i, j), (i, j + 1))
              else:
                G.add_edge((i, j + 1), (i, j))

            # Conexión con el vecino de abajo (bidireccional)
            if i < n - 1:
              # la direccion depende de la paridad de j respecto a d
              if j%(2*d) == 0:
                G.add_edge((i + 1, j), (i, j))
              else:
                G.add_edge((i, j), (i + 1, j))


    # quitar nodos
    nodos_quitar = [(x, y) for (x, y) in G.nodes() if x%d != 0 and y%d != 0]
    G.remove_nodes_from(nodos_quitar)

    # definir el pos
    pos = {(x,y):(y,-x) for x,y in G.nodes()}

    return G, pos



# funcion para hacer un grafo personalizado
def grafo_personalizado(lado, d, nodos_mitad, bidireccion = False):
  '''
  lado - lados del rectangulo
  d - cada cuantos nodos poner una calle
  nodos_mitad - cuantos nodos en cada diagonal
  '''

  # Crear el grafo no dirigido
  G0 = nx.grid_graph(dim=[lado, lado])
  # Crear el dirigido y llenarlo
  G = nx.DiGraph()
  G.add_nodes_from(G0.nodes())
  for u, v in G0.edges():
      # u y v son tuplas representando las coordenadas (x, y)
      if v[0] == u[0] and v[1] == u[1] + 1:     # Movimiento hacia la derecha
          G.add_edge(u, v)
      elif v[0] == u[0] + 1 and v[1] == u[1]:  # Movimiento hacia abajo
          G.add_edge(u, v)

  # quitar nodos
  nodos_quitar = [(x, y) for (x, y) in G.nodes() if x%d != 0 and y%d != 0]
  G.remove_nodes_from(nodos_quitar)


  # agregar aristas regreso
  # elementos diagonal
  diagonal = [x for x in range(lado) if x%d==0]
  # conectar por cada diagonal
  for idx_diag in range(1, len(diagonal)):
    # poner del existente al primero
    G.add_node((diagonal[idx_diag] - d/nodos_mitad, diagonal[idx_diag] - d/nodos_mitad))
    G.add_edge((diagonal[idx_diag], diagonal[idx_diag]), (diagonal[idx_diag] - d/nodos_mitad, diagonal[idx_diag] - d/nodos_mitad))
    # poner del primero al penultimo
    for idx_mitad in range(1, nodos_mitad):
      G.add_node((diagonal[idx_diag] - (idx_mitad+1)*d/nodos_mitad, diagonal[idx_diag] - (idx_mitad + 1)*d/nodos_mitad))
      G.add_edge((diagonal[idx_diag] - (idx_mitad)*d/nodos_mitad, diagonal[idx_diag] - (idx_mitad)*d/nodos_mitad), (diagonal[idx_diag] - (idx_mitad+1)*d/nodos_mitad, diagonal[idx_diag] - (idx_mitad+1)*d/nodos_mitad))
    # poner del penultimo al ultimo
    G.add_edge((diagonal[idx_diag] - (nodos_mitad - 1)*d/nodos_mitad, diagonal[idx_diag] - (nodos_mitad - 1)*d/nodos_mitad), (diagonal[idx_diag - 1], diagonal[idx_diag - 1]))


  # si se quiere, poner todas las aristas en direccion contraria
  if bidireccion:
    for u, v in G.edges():
      G.add_edge(v, u)

  # definir el pos
  pos = {(x,y):(y,-x) for x,y in G.nodes()}

  return G, pos



# funciones para el data collector

# numero de agentes
def contar_agentes(modelo):
    """
    Esta función se usa para contar el número de agentes en el modelo.
    """
    return modelo.schedule.get_agent_count()

# numero de agentes esperando
def contar_agentes_esperando(modelo):
  """
  Esta función se usa para contar el número de agentes que se encuentran esperando
  Es decir, su ultimo movimiento fue quedarse en el mismo nodo donde estaban
  """
  return len([agent for agent in modelo.schedule.agents_by_type[Coche].values() if agent.esperando])
