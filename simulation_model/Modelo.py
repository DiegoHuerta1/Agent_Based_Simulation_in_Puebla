# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:24:17 2024

@author: diego
"""


import mesa


import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import pickle
from matplotlib.ticker import MaxNLocator
from collections import Counter
import matplotlib.colors as mcolors
import os
import copy
import gc
from IPython.display import clear_output
import matplotlib.pyplot as plt



from .funciones_auxiliares import precomputar_distancias_,networkx_2_igraph, contar_agentes, contar_agentes_esperando, get_color_aleatorio
from .Agentes import Coche


'''
Clase del modelo
Controla toda la simulacion
'''




class Modelo(mesa.Model):
  '''
  Clase del modelo
  Controla el centro historico, y como se comportan los coches dentro de el
  '''

  # constructor
  def __init__(self, info_grafo, argumentos_visualizacion):

    '''
    Para el constructor se especifica:
      info_grafo = diccionario con la informacion del grafo
        G - grafo
        pos - posicion de cada nodo
        proba_origen - probabilidad de cada nodo de ser un origen
        proba_destino - probabilidad de cada nodo de ser destino
     argumentos_visualizacion - varios parametros que personalizan la visualizacion
    '''
    # poner atributos del grafo
    self.G = copy.deepcopy(info_grafo["G"]) # copiarlo para no agregarle atributos ni nada
    self.num_nodos = self.G.number_of_nodes()
    self.pos = info_grafo["pos"]
    self.proba_origen = info_grafo["proba_origen"]
    self.proba_destino = info_grafo["proba_destino"]
    # guardar otra copia del grafo, que no se modifique (nunca se ponen agentes ahi)
    self.G_limpio = copy.deepcopy(info_grafo["G"])
    # de momento quiero hacer swap
    self.swap = True

    # usar el grafo de nx y el pos para tener distancias en el plano
    self.pares_distancias_plano = precomputar_distancias_(self.G, self.pos)

    # los calculos se hacen en un gafo de igraph
    # transformar el grafo de nx a igraph
    info_G_igraph = networkx_2_igraph(self.G)
    # poner como atributos los datos de la conversion
    self.G_igraph = info_G_igraph["G_igraph"]
    self.dict_nodes_2_idx = info_G_igraph["nodes_2_idx"]
    self.dict_idx_2_nodes = info_G_igraph["idx_2_nodes"]
    # pesos en las aristas que son distancias en el plano
    self.edges_weight_dist = [self.obtener_distancia_plano(None, edge.source, edge.target) for edge in self.G_igraph.es]
    # iniciar con estos pesos, hasta que se actualizen y se considere trafico
    self.G_igraph.es['weights'] = self.edges_weight_dist

    # poner atributos de visualizacion si es que estan en el diccionario
    self.node_size = argumentos_visualizacion.get("node_size", 10)
    self.edge_color = argumentos_visualizacion.get("edge_color", "gray")
    self.arrowsize = argumentos_visualizacion.get("arrowsize", 10)
    self.color_vacio = argumentos_visualizacion.get("color_vacio", '#f0f0f0')
    self.color_bloqueo = argumentos_visualizacion.get("color_bloqueo", 'red')
    self.color_atasco = argumentos_visualizacion.get("color_atasco", 'red')

    # idenfiticar las aristas con doble direccione
    self.bidirectional_edges = []
    for u, v in self.G.edges():
      if self.G.has_edge(v, u):
        if (v, u) not in self.bidirectional_edges:
            self.bidirectional_edges.append((u, v))


    # indicar que el grafo es el entorno
    self.grid = mesa.space.NetworkGrid(self.G)

    # iniciar el scheduler
    self.schedule = mesa.time.RandomActivationByType(self)
    self.running = True

    # iniciar data collector
    self.datacollector = mesa.DataCollector(
      # informacion a nivel de modelo
      model_reporters = {
          "Numero de agentes": contar_agentes,
          "Agentes esperando": contar_agentes_esperando
      }
    )

    # diccionario donde guardo a todos los agentes
    self.historico_agentes = dict()
    # id de los agentes creados
    self.agent_id = 0

  # end constructor

  # -------------------------------------------------------------------------------------------

  # Pesos en las aristas del grafo de igraph para calcular caminos cortos en el grafo

  # distancia en el plano entre dos nodos
  def obtener_distancia_plano(self, arg_grafo, u, v):
    '''
    Obtener la distancia en el plano entre dos nodos (usando el pos)
    Argumentos:
      arg_grafo - no se usa
      u - nodo 1, se usa el indice del nodo, pues asi se trabaja en igraph (u = self.dict_nodes_2_idx[nodo_u])
      v - nodo 2, se usa el indice del nodo, pues asi se trabaja en igraph (v = self.dict_nodes_2_idx[nodo_v])
    '''
    # si u == v, entonces su distancia es 0
    if u == v:
      return 0

    # se toman indices i,j de manera que i<j
    i = min(u, v)
    j = max(u, v)

    # devolver la distancia, ya se tienen calculadas
    # ver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    return self.pares_distancias_plano[self.num_nodos * i + j - ((i + 2) * (i + 1)) // 2]

  # calcular trafico de una arista
  def calcular_trafico(self, u, v):
    '''
    Calcular el trafico de una arista
    Argumentos:
      u - nodo 1, se usa el indice del nodo, pues asi se trabaja en igraph (u = self.dict_nodes_2_idx[nodo_u])
      v - nodo 2, se usa el indice del nodo, pues asi se trabaja en igraph (v = self.dict_nodes_2_idx[nodo_v])

    Devuelve el peso que indica el trafico que hay en la arista,
    este peso de trafico se va a multiplicar por el peso de la arista para obtener el peso final
    '''

    # el trafico de la arista depende de la opcion que se haya seleccionado

    # null - la precencia de coches no aumenta el peso de una arista
    if self.pesos_trafico == "null":
      return 1

    # simple - si la arista tiene 1 coche, su peso se multiplica por 1.5, si tiene 2 se multiplica por 2
    elif self.pesos_trafico == "simple":
      return 2 - (self.grid.is_cell_empty(self.dict_idx_2_nodes[u])* 0.5) - (self.grid.is_cell_empty(self.dict_idx_2_nodes[v])* 0.5)



  # poner pesos en las aristas del grafo
  def update_edges_weights(self):
    '''
    Actualizar los pesos en las aristas del grafo de igraph

    Los pesos (w:E -> R) se calculan como:
    w(u, v) = distancia(u, v) * trafico(u, v) * bloqueo(u, v)

    Donde
    distancia(u, v) es distancia en el plano entre u y v
    trafico(u, v) aumenta segun los cohces presentes en esa arista
    bloqueo(u, v) es infinito si la arista esta bloqueda actualmente
    '''

    # calcular distintos pesos:

    # distancia (solo es tomarlo, ya estan calculados)
    pesos_distancia = np.array(self.edges_weight_dist)
    # trafico (calcular el trafico de cada arista)
    pesos_trafico = np.array([self.calcular_trafico(edge.source, edge.target) for edge in self.G_igraph.es])
    # bloqueos
    pesos_bloqueo =  np.ones_like(pesos_trafico) # primero puros 1
    if len(self.aristas_bloqueadas_actuales) > 0:
      pesos_bloqueo[self.aristas_bloqueadas_actuales] = np.inf # poner infinito en las aristas bloqueadas

    # poner los pesos finales
    self.G_igraph.es['weights'] = pesos_distancia*pesos_trafico*pesos_bloqueo


  # distancia en el grafo entre nodos
  def shortest_path_graph(self, nodo_from, nodo_to):
    '''
    Devuelve el camino mas corto del nodo_from al nodo_to
    Se usa el grafo de igraph, pues es mas eficiente
    Se utilizan pesos en las aristas, obviamente se usan las de la ultima actualizacion
    '''

    # calcular el camino mas corto con los pesos, pasar los nodos a sus indices, pues se calcula en igraph
    shortest_path_ig = self.G_igraph.get_shortest_path_astar(v = self.dict_nodes_2_idx[nodo_from],
                                                             to = self.dict_nodes_2_idx[nodo_to],
                                                             weights= 'weights',
                                                             mode = 'out',
                                                             heuristics = self.obtener_distancia_plano,
                                                             output = "vpath")

    # convertir el camino a nodos del grafo, no sus indices
    shortest_path = [self.dict_idx_2_nodes[node] for node in shortest_path_ig]
    return shortest_path


  # distancias en el grafo vacio
  def calcular_tiempo_minimo(self, origen, destino):
    '''
    Devuelve el tiempo minimo que se puede tomar un agente
    en ir desde un origen hasta un destino
    Entonces calcula el camino con menos nodos posibles, y devuelve la longuitud
    '''

    # calcular el camino mas corto en termino de menos nodos
    shortest_distance = self.G_igraph.distances(source = [self.dict_nodes_2_idx[origen]],
                                                target = [self.dict_nodes_2_idx[destino]],
                                                weights = None, # no considerar pesos, solo menos nodos en el camino
                                                mode = 'out')
    # solo devovler la distancia
    return shortest_distance[0][0]


  # calcular tiempos minimo posible todos los agentes
  def calcular_tiempos_minimos_agentes(self, ver_tqdm = False):
    '''
    Para cada agente del historico, calcular el tiempo minimo posible
    que se toma de ir desde el origen al destino
    '''

    # tomar los agentes
    agentes_iterar = self.historico_agentes.values()
    # ver si se va a ver progreso
    if ver_tqdm:
      agentes_iterar = tqdm(agentes_iterar, desc="Tiempos minimos:")

    # iterar en todos los agentes que se crearon
    for coche in agentes_iterar:
      # calcular su tiempo minimo
      coche.tiempo_minimo = self.calcular_tiempo_minimo(coche.origen, coche.destino)

  # ---------------------------------------------------------------------------

  # bloqueos

  # agregar aristas bloqueadas de ig
  def extender_bloqueos_con_aristas(self):
    '''
    Por cada bloqueo en self.bloqueos,
    agregar como informacion del bloqueo
    las aristas que son bloqueadas (usando los indices del grafo de igraph)
    '''

    # iterar en cada bloqueo
    for bloqueo in self.bloqueos:

      # tomar los nodos bloqueados
      nodos_bloqueados = bloqueo["nodos_bloqueados"]

      # tomar todas las aristas del grafo que llegan a un nodo bloqueado
      aristas_bloqueo = [(u, v) for (u, v) in self.G.edges() if v in nodos_bloqueados]

      # convertir a las aristas a sus indices en el grafo de igraph
      aristas_igraph_bloqueo = [self.G_igraph.get_eid(self.dict_nodes_2_idx[u],
                                                      self.dict_nodes_2_idx[v])
                                for (u, v) in aristas_bloqueo]

      # agregar la info de las aristas al bloqueo
      bloqueo["aristas_igraph_bloqueadas"] = aristas_igraph_bloqueo


  # actualizar segun el step
  def actualizar_bloqueos(self):
    '''
    Actualizar la lista de bloqueos actuales y los nodos bloqueados, segun el step actual
    '''

    # iterar en todos los bloqueos, y dejar los que sean vigentes
    self.bloqueos_actuales = [bloqueo for bloqueo in self.bloqueos
                              if bloqueo["inicio_bloqueo"] <= self.schedule.steps
                              and bloqueo["fin_bloqueo"] >= self.schedule.steps]

    # juntar todos los nodos y aristas bloqueados
    self.nodos_bloqueados_actuales = set()
    self.aristas_bloqueadas_actuales = []
    for bloqueo in self.bloqueos_actuales:
      self.nodos_bloqueados_actuales.update(bloqueo["nodos_bloqueados"])
      self.aristas_bloqueadas_actuales.extend(bloqueo["aristas_igraph_bloqueadas"])

  # -------------------------------------------------------------------------------------------

  # Creacion de agentes

  # seleccionar para cada agente, donde inicia y termina
  def get_origen_destino_agente(self):
    '''
    Devuelve un origen y destino para un agente
    Se construye de modo que se puede llegar del origen al destino, el origen es vacio
    ademas, estos son aleatorios de acuerdo a las probabilidadas dadas
    '''

    # Seleccionar origen de acuerdo a las probabilidades, condicional a vacio y no boqueado

    # solo los nodos vacios no bloqueados pueden ser origenes
    posibles_origenes = [v for v in self.G.nodes()
                         if self.grid.is_cell_empty(v)
                         and v not in self.nodos_bloqueados_actuales]

    # inicar vacio
    posibles_destinos = []

    # seleccionar origenes hasta que se tenga uno que tenga destinos disponibles
    while len(posibles_origenes) > 0 and len(posibles_destinos) == 0:

      # tomar un origen aleatorio y quitarlo de la lista
      pesos = np.array([self.proba_origen[nodo] for nodo in posibles_origenes])
      indice_seleccionado = np.random.choice(len(posibles_origenes), p =  pesos/pesos.sum())
      origen_seleccionado = posibles_origenes.pop(indice_seleccionado)

      # ver los posibles destinos de este origen
      # recordar no considerar pasar por bloqueados

      # Crear subgrafo sin nodos bloqueados y obtener su correspondencia
      nodos_permitidos = [self.dict_nodes_2_idx[v] for v in self.G.nodes() if v not in self.nodos_bloqueados_actuales]
      G_sin_bloqueados = self.G_igraph.subgraph(nodos_permitidos)

      # Obtener los alcanzables en el subgrafo
      nodo_origen_subgrafo = nodos_permitidos.index(self.dict_nodes_2_idx[origen_seleccionado])
      nodos_alcanzables_filtrados_subgrafo = G_sin_bloqueados.neighborhood(
                                                                    vertices= nodo_origen_subgrafo,
                                                                    order=1000000,
                                                                    mode='out',
                                                                    mindist=1
      )

      # Convertir los índices del subgrafo a los índices originales de igraph
      nodos_alcanzables_igraph = [nodos_permitidos[n] for n in nodos_alcanzables_filtrados_subgrafo]
      # transormar a nx
      nodos_alcanzables = [self.dict_idx_2_nodes[node] for node in nodos_alcanzables_igraph]
      posibles_destinos = [v for v in nodos_alcanzables
                           if v not in self.nodos_bloqueados_actuales]
      # filtrar solo con proba positiva
      posibles_destinos = [v for v in posibles_destinos if self.proba_destino[v] > 0]

    # end while
    # a este punto, o ya no hay origenes posibles, o ya se tienen destinos poisbles

    # si no hay origenes posibles (grafo lleno) mandar error
    if len(posibles_origenes) == 0:
      raise Exception("No hay origenes posibles que tengan destinos posibles")

    # Seleccionar destino de acuerdo a las probabilidades
    pesos = np.array([self.proba_destino[nodo] for nodo in posibles_destinos])
    # remplazar Nan por 0
    pesos[np.isnan(pesos)] = 0
    # tomar un destino aleatorio
    indice_seleccionado = np.random.choice(len(posibles_destinos), p =  pesos/pesos.sum())
    destino_seleccionado = posibles_destinos[indice_seleccionado]

    # devolver el par
    return origen_seleccionado, destino_seleccionado


  # funcion para crear agentes
  def create_agents(self, num):
    '''
    Crear agentes con origenes (vacios) y destinos aleatorios
    ponerles id unico, ponerlos en el grafo, y en el historico de agentes
    '''

    # iterar segun todos los agentes que se quieran
    for _ in range(num):

      # tomar un color aleatorio para este agente
      color_agente = get_color_aleatorio()
      # tomar el origen y el destino del agente
      origen, destino = self.get_origen_destino_agente()

      # crear agente, el step es el del schedule
      coche = Coche(self.agent_id, self, origen, destino, color_agente,
                    step_creation = self.schedule.steps,
                    tiempo_recalcular = self.tiempo_recalcular_agentes)

      # meter a un diccionario donde se guardan agentes
      self.historico_agentes[self.agent_id] = coche
      # incrementar el id de los agentes para tener unicos
      self.agent_id += 1

    # end for agentes
  # end create_agents

  # -------------------------------------------------------------------------------------------

  # lista de agentes tipo coche
  def get_coches(self):
    '''
    Devuelve un iterador de los agentes del modelo tipo coche
    El orden es aleatorio en cada iteracion
    '''
    # lista de coches
    coches = list(self.schedule.agents_by_type[Coche].values())
    # revolver
    self.random.shuffle(coches)
    return coches


  # delimitar cuandos agentes se crean
  def get_number_new_agents(self):
    '''
    Obtiene el numero de agentes de crear en un step

    Este numero es aleatorio, y depende del step en el que se esté,
    por medio de la funcion que da el parametro lambda para cada step
    '''

    # primero, usar la funcion delimitada para tomar el parametro
    if self.lambda_tiempo is not None:
      # tomar el parametro
      lambda_step = self.lambda_tiempo(self.schedule.steps)
    # si no hay funcion, pues tomar parametro 1
    else:
      print("WARNING: No hay funcion para delimitar el parametro lambda")
      lambda_step = 1

    # samplear usando el parametro
    return np.random.poisson(lambda_step)

  # -------------------------------------------------------------------------------------------

  ####################################
  ############## STEP ################
  ####################################
  # step function del modelo
  def step(self, generar = True):
    '''
    Ejecuta un paso del modelo, acciones de un step
    Esta funcion debe de llamarse unicamente al correr el modelo, no de forma individual

    1) Se quitan agentes que ya esten en su destino
    2) Se quitan los agentes que esten o quieran llegar a un nodo que va a ser bloqueado este step
       (estos nodos tambien se quitan del historico, totalmente se elimina rastro de su existencia)
    3) Se actualizan los pesos en las aristas dependiendo el trafico actual y los bloqueos
    4) Si hubo bloqueos nuevos en esta step, todos los agentes recalculan ruta
    5) Se mueven los agentes
    6) Hacer swap en aristas bidireccionales (si se quiere)
    7) Se crean nuevos agentes (opcional, parametro generar)
    '''

    # 1) Se quitan agentes que ya esten en su destino

    # tomar los coches del modelo
    coches_modelo = self.get_coches()
    # iterar en estos
    for coche in coches_modelo:
      # ver si ya llego a su destino
      coche.maybe_terminar()


    # 2) Se quitan los agentes que esten o quieran llegar a un nodo que va a ser bloqueado este step

    # primero ver los bloqueos que comienzan en este step
    bloqueos_nuevos = [bloqueo for bloqueo in self.bloqueos_actuales
                        if bloqueo["inicio_bloqueo"] == self.schedule.steps]

    # si es que en este step inicia un bloqueo
    if len(bloqueos_nuevos) > 0:

      # tomar todos los nodos que son bloqueados por estos bloqueos
      nodos_bloqueados_nuevos = set()
      for bloqueo in bloqueos_nuevos:
        nodos_bloqueados_nuevos.update(bloqueo["nodos_bloqueados"])

      # tomar todos los coches que esten ahi, o todos los que quieran llegar a un nodo bloqueado
      agentes_bloqueados = [coche for coche in self.schedule.agents_by_type[Coche].values()
                            if coche.node in nodos_bloqueados_nuevos or coche.destino in nodos_bloqueados_nuevos]

      # eliminar estos cohces del grafo, del scheduler y del historico de agentes (como si nunca estuvieron vivos)
      for agente_eliminar in agentes_bloqueados:
        self.grid.remove_agent(agente_eliminar)
        self.schedule.remove(agente_eliminar)
        del self.historico_agentes[agente_eliminar.unique_id]
    # end if - iniciar un bloqueo nuevo


    # 3) Se actualizan los pesos de acuerdo a trafico y bloqueos actuales
    self.update_edges_weights()


    # 4) Si hubo bloqueos nuevos en esta step, todos los agentes recalculan ruta
    
    # se eliminan los coches que ya no puedan llegar a su destino por algun bloqueo
    coches_eliminar = []

    # solo si hubo bloqueos nuevos
    if len(bloqueos_nuevos) > 0:

      # para todo agente
      for coche in self.schedule.agents_by_type[Coche].values():
        # recalcular
        coche.recalcular_camino()

        # ver si se tiene que eliminar

        # si ya no pueden llegar a su destino
        # (talvez un bloqueo hizo que ya no puedan llegar)
        if coche.destino not in coche.next_steps:
          # borrar
          coches_eliminar.append(coche)

        # otra posibilidad en la que se elimina
        else: 
          # si tienen que pasar por el bloqueo
          for nodo_prohibido in nodos_bloqueados_nuevos:
            # si esta en el camino
            if nodo_prohibido in coche.path:
              # borrar
              coches_eliminar.append(coche)
              break
            # end if nodo prohibido en path
          # end for en nodo prohibido
        # end else 
      # end for agentes

      # eliminar los agentes necesarios
      for agente_eliminar in coches_eliminar:
        self.grid.remove_agent(agente_eliminar)
        self.schedule.remove(agente_eliminar)
        del self.historico_agentes[agente_eliminar.unique_id]

    # end if - hay bloqueos nuevos


    # 5) Se mueven los agentes

    # tomar todos los coches
    coches_modelo = self.get_coches()
    for coche in coches_modelo:
      # hacer el paso del agente
      coche.step()


    # 6) Hacer swap en aristas bidireccionales
    # Si en este step dos agentes no pudieron mover por aristas bidireccionales, hacer swap

    # si es que se quiere
    if self.swap:

      # iterar en las aristas bidireccionales
      for (u, v) in self.bidirectional_edges:

        # ver si hay un agente en los nodos
        contenido_u = self.grid.get_cell_list_contents([u])
        contentido_v = self.grid.get_cell_list_contents([v])

        # si una es vacia, ya no se hace nada
        if len(contenido_u) == 0 or len(contentido_v) == 0:
          continue

        # las dos tienen agentes
        # comprobar que hay solo un coche
        assert len(contenido_u) == 1
        assert len(contentido_v) == 1
        # tomar el agente de cada celda
        agente_u = contenido_u[0]
        agente_v = contentido_v[0]

        # si no se cumple la condicion del swap, pues no se hace
        if not (agente_u.esperando and agente_v.esperando and agente_u.next_steps[0] == v and agente_v.next_steps[0] == u):
          continue

        # ambos agentes estan esperando
        # y quieren ir al otro nodo

        # Hacer swap

        # hacer como que la ultima espera no ocurrio
        agente_u.esperando = False
        agente_v.esperando = False
        # tambien modificar su camino (quitar el ultimo nodo)
        agente_u.path = agente_u.path[:-1]
        agente_v.path = agente_v.path[:-1]
        # quitar ese nodo de los proximos pasos
        agente_u.next_steps.pop(0)
        agente_v.next_steps.pop(0)
        # ahora si modificar su posicion
        agente_u.mover(v)
        agente_v.mover(u)

      # end for aristas bidireccionales
    # end if - swap


    # 7) Se crean nuevos agentes

    # si es que se quiere generar
    if generar:
      # delimitar cuantos agentes se crean (es aleatorio y depende del step)
      num_nuevos_coches = self.get_number_new_agents()
      # crear los agentes
      self.create_agents(num = num_nuevos_coches)

    # indicar que termina un step
    self.schedule.steps += 1
    # recolectar los datos del modelo en este step
    self.datacollector.collect(self)

  # end step function

  # -------------------------------------------------------------------------------------------
  # Correr el modelo

  ####################################
  ############## RUN #################
  ####################################
  def run_model(self, poblacion_inicial = 10,
                lambda_tiempo = None,
                steps_generando = 10, steps_finales = 10,
                visualizar = True, sleep = 0.5,
                ver_barra_progreso = False,
                pesos_trafico = "simple",
                tiempo_recalcular_agentes = 10,
                bloqueos = []):
    '''
    Correr el modelo, parametros:
      poblacion_inicial = numero de agentes iniciales (iteracion 0)
      lambda_tiempo = funcion que da el parametro lambda para generar agentes en cada step
      steps_generando = numero de steps en los que se generan agentes
      steps_finales = numero de steps en los que no se generan agentes
      visualizar = True si se quiere ver el grafo en cada step
      sleep = tiempo de espera entre steps si se esta visualizando
      ver_barra_progreso = True si se quiere ver una barra de progreso de los steps
      pesos_trafico = indica la forma en que la precencia de coches aumenta el peso en las aristas
                null - la precencia de coches no aumenta el peso de una arista
                simple - si la arista tiene 1 coche, su peso se multiplica por 1.5, si tiene 2 se multiplica por 2
      tiempo_recalcular_agentes = que tan seguido un agente recalcula la ruta preferida
      bloqueos - una lista de bloqueos en la red que se van a tener en la simulacion, cada bloqueo es un diccionarion con
                "inicio_bloqueo" - step en el que se inicia este bloqueo
                "fin_bloqueo" - step en el que se termina este bloqueo
                "nodos_bloqueados" - conjunto de nodos que se van a bloquear
    '''

    # poner como atributo la manera en que el trafico afecta a los pesos, el tiempo de recalcular y los bloqueos
    self.pesos_trafico = pesos_trafico
    self.tiempo_recalcular_agentes = tiempo_recalcular_agentes
    self.bloqueos = bloqueos
    self.extender_bloqueos_con_aristas() # por cada bloqueo, saber que aristas bloquea
    self.actualizar_bloqueos()  # mantener una lista de bloqueos y de nodos bloqueados actuales para cada step
    self.lambda_tiempo = lambda_tiempo # funcion que da el parametro lambda para generar agentes en cada step

    # Primero hacer la poblacion inicial
    self.create_agents(num = poblacion_inicial)
    # aca se marca un step, se considera que la poblacion inicial se crea en el step 0,
    # despues inician los steps regulares, iniciando la cuenta en 1
    self.schedule.steps += 1
    # tambien indicarlo en el data collector
    self.datacollector.collect(self)

    # si se quiere, visualizar despues del step 0
    if visualizar:
      self.visualizar_estado_modelo(sleep = sleep, limite_x = steps_generando + steps_finales)


    # delimitar el numer de steps a hacer, y si se va a ver progreso
    total_steps = steps_generando + steps_finales
    rango_iterable = range(1, 1 + total_steps)
    if ver_barra_progreso:
        rango_iterable = tqdm(rango_iterable, desc="Steps maximos:")

    # hacer los steps que se quieran
    for _ in rango_iterable:

      # segun el step en el que se esta, ver si se generan agentes
      generar_step = (self.schedule.steps <= steps_generando)
      # actualizar la lista de bloqueos actuales para este step
      self.actualizar_bloqueos()
      # hacer el step
      self.step(generar = generar_step)

      # si se quiere, visualizar
      if visualizar:
        self.visualizar_estado_modelo(sleep = sleep, limite_x = steps_generando + steps_finales)

      # al terminar el step, terminar si
      # si ya no estamos generando agentes, y ya no hay en el mapa
      if not generar_step and self.schedule.get_agent_count() == 0:
        break

    # end for steps

    # despues de hacer todos los steps,
    # calcular el tiempo minimo de todos los agentes
    self.calcular_tiempos_minimos_agentes(ver_tqdm = ver_barra_progreso)

  # end function run_model


  # --------------------------------------------------------------------------

  # Visualizaciones

  # funcion para visualizar el modelo
  def visualizar_estado_modelo(self, sleep, limite_x):
    '''
    Visualiza el estado actual del modelo
    Muestra el mapa con los agentes y otras graficas
    Hace la grafica, se espera un tiempo de sleep,
    y finalmente limpia la figura
    '''

    # hacer la figura, es un mosaico
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    axd = fig.subplot_mosaic([["Map", "Up"],
                              ["Map", "Down"]])

    # sacar datos del data collector para visualizar
    df_datos = self.datacollector.get_model_vars_dataframe()

    # 1) Hacer el mapa
    self.dibujar_grafo(ax=axd["Map"])
    axd["Map"].set_title("Mapa")

    # 2) Mostrar el numero de agentes actualmente
    axd["Up"].plot(np.array(df_datos.index), df_datos["Numero de agentes"])
    axd["Up"].set_xlabel("Step")
    axd["Up"].set_ylabel("Coches")
    axd["Up"].set_title("Numero de agentes en el mapa")
    axd["Up"].set_xlim(0, limite_x) # mismo limite siempre
    axd["Up"].set_ylim(bottom=0)  # eje y comienza en 0
    axd["Up"].spines['top'].set_visible(False)  # no ver linea
    axd["Up"].spines['right'].set_visible(False) # no ver linea
    axd["Up"].xaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x
    axd["Up"].yaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x

    # # 3) Mostrar el numero de agentes que estan esperando actualmente
    axd["Down"].plot(np.array(df_datos.index), df_datos["Agentes esperando"])
    axd["Down"].set_xlabel("Step")
    axd["Down"].set_ylabel("Coches")
    axd["Down"].set_title("Numero de agentes esperando")
    axd["Down"].set_xlim(0, limite_x) # mismo limite siempre
    axd["Down"].set_ylim(bottom=0)  # eje y comienza en 0
    axd["Down"].spines['top'].set_visible(False)  # no ver linea
    axd["Down"].spines['right'].set_visible(False) # no ver linea
    axd["Down"].xaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x
    axd["Down"].yaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x

    # titulo y mostrar
    fig.suptitle(f"Step: {self.schedule.steps - 1}") # -1 porque se visualiza despues de hacer el step
    plt.show()

    # esperar mientras se contempla este grafo
    time.sleep(sleep)
    # finalmente limpiar
    clear_output(wait=True)
  # end function visualizar modelo


  # funcion para dibujar el grafo
  def dibujar_grafo(self, ax = None):
    '''
    Dibuja el grafo
    El color de una celda depende de si esta vacia o no
    El color de un nodo depede de si esta bloqueado o no, o de si hay agente
    '''

    # Definir colores, segun los colores de los agentes


    colores_nodos = []
    # iterar nodos
    for nodo in self.G.nodes():
      #
      # agente
      # bloqueo
      # vacio

      # si es bloqueado va de color especial
      if nodo in self.nodos_bloqueados_actuales:
        colores_nodos.append(self.color_bloqueo)
      # no es bloqueo, y esta vacio, va de un color especial
      elif self.grid.is_cell_empty(nodo):
        colores_nodos.append(self.color_vacio)
      # no esta vacio, tomar el color de agente que esta ahi
      else:
        # tomar contendio
        contenido_nodo = self.grid.get_cell_list_contents([nodo])
        # comprobar que sea soolo un agente
        assert len(contenido_nodo) == 1
        # agregar el color de este agente
        colores_nodos.append(contenido_nodo[0].color)

    # Hacer la figura

    # si no se tiene ax hacer uno
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(5, 5))

    # dibujar en el ax
    nx.draw(self.G, self.pos, node_color = colores_nodos, edge_color = self.edge_color,
            node_size = self.node_size, arrowsize = self.arrowsize, ax=ax)
  # end dibujar_grafo




  # numero de agentes
  def mostrar_agentes_presentes(self, ax=None):
    '''
    Muestra los agentes en el mapa como funcion del tiempo
    '''
      
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(10, 4))
        
    # tomar df de data collectos
    results = self.datacollector.get_model_vars_dataframe()
    
    # graficar
    ax.plot(np.array(results.index), results["Numero de agentes"]) # +1 para que inicie en 0
    ax.set_xlabel("Step")
    ax.set_ylabel("Coches")
    ax.set_title("Numero de agentes en el mapa")
    ax.set_ylim(bottom=0)  # eje y comienza en 0
    ax.spines['top'].set_visible(False)  # no ver linea
    ax.spines['right'].set_visible(False) # no ver linea
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x

    

  # numero de agentes esperando
  def mostrar_agentes_esperando(self, ax=None):
    '''
    Muestra los agentes que estan esperando en el mapa como funcion del tiempo
    '''
      
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(10, 4))
        
    # tomar df de data collectos
    results = self.datacollector.get_model_vars_dataframe()
    
    # graficar
    ax.plot(np.array(results.index), results["Agentes esperando"]) # +1 para que inicie en 0
    ax.set_xlabel("Step")
    ax.set_ylabel("Coches")
    ax.set_title("Numero de agentes esperando")
    ax.set_ylim(bottom=0)  # eje y comienza en 0
    ax.spines['top'].set_visible(False)  # no ver linea
    ax.spines['right'].set_visible(False) # no ver linea
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # solo ticks enteros en x



  # dibujar solo un camino
  def dibujar_path(self, path, color_path = "green", alpha_min = 0.3, ax=None):
    '''
    Dibuja un camino en el grafo
    '''

    # hacer rgb el color
    color_path = mcolors.to_rgba(color_path)
    # si no hay ax hacer uno
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(8, 8))

    # sacar aristas del path
    path_edges = list(zip(path, path[1:]))


    # frecuencias de nodos entre alpha_min y 1 para controlar intensidad
    frecuencias_nodos = Counter(path)
    frecuencias_normalizadas = np.array([frecuencias_nodos.get(nodo, 0) for nodo in self.G.nodes()])
    frecuencias_normalizadas = frecuencias_normalizadas / frecuencias_normalizadas.max()
    frecuencias_con_alfa_min = alpha_min + (1 - alpha_min) * frecuencias_normalizadas

    # delimitar colores de los nodos y aristas, segun si estan en el path
    colores_nodos = [(color_path[0], color_path[1], color_path[2], frecuencia)
                      if nodo in path else self.color_vacio
                      for nodo, frecuencia in zip(self.G.nodes(), frecuencias_con_alfa_min)]
    colores_aristas = [color_path if edge in path_edges else self.edge_color
                       for edge in self.G.edges()]

    # delimitar tamaños segun si estan en el path, y su frecuencia
    tamaños_nodos = [frecuencia*self.node_size for frecuencia in (1+frecuencias_normalizadas)]

    # dibujar
    nx.draw(self.G, self.pos, node_color = colores_nodos, edge_color = colores_aristas,
            node_size = tamaños_nodos, arrowsize = self.arrowsize, ax=ax)

  # end dibujar_path


  # obtener centralidades de los nodos y aristas
  def obtener_centralidades_nodos_aristas(self):
    '''
    Calcula centralidades de los nodos y aristas, dados por las frecuencias en la simulacion
    '''

    # juntar todos los caminos de todos los agentes (juntar nodos y aristas)
    nodos_all_paths = []
    edges_all_paths = []
    # iterar en los agentes
    for agent in self.historico_agentes.values():
      # agregar los nodos
      nodos_all_paths += agent.path
      # agregar las aristas
      edges_all_paths += list(zip(agent.path, agent.path[1:]))

    # sacar centralidades, que son frecuencias

    # nodos
    frecuencias_nodos = Counter(nodos_all_paths)
    centralidades_frecuencias_nodos = np.array([frecuencias_nodos.get(nodo, 0) for nodo in self.G.nodes()])
    # edges
    frecuencias_edges = Counter(edges_all_paths)
    centralidades_frecuencias_edges = np.array([frecuencias_edges.get(edge, 0) for edge in self.G.edges()])

    # devolver
    return centralidades_frecuencias_nodos, centralidades_frecuencias_edges


  # ver lo de todos los caminos juntos
  def dibujar_centralidades(self, alpha_min = 0.1, ax=None):
    '''
    Dibuja todo el grafo, denotando la centralidad de simulacion de cada nodo 
    en los caminos que toman los agentes
    '''

    # obtener centralidades de nodos y aristas
    centr_nodos, centr_edges = self.obtener_centralidades_nodos_aristas()

    # si no hay ax hacer uno
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(8, 8))


    # hacer min-max scaling de las centralidades de los nodos
    centr_nodos_escalado = (centr_nodos - centr_nodos.min()) / (centr_nodos.max() - centr_nodos.min())
    # delimitar tamaños de nodos segun su centralidad escalada
    tamaños_nodos = (2*centr_nodos_escalado+0.5)*self.node_size

    # dibujar
    nx.draw(self.G, self.pos, node_color = centr_nodos, edge_color = centr_edges,
            node_size = tamaños_nodos, arrowsize = self.arrowsize, ax=ax)

  # end dibujar_frecuencias


  # obtener centralidades de los nodos y aristas como un df
  def obtener_centralidades_df(self):
    '''
    Calcula centralidades de los nodos y aristas, dados por las frecuencias en la simulacion
    Devuelve los resultados en dos df, uno para ndoos, otro para aristas
    '''

    # obtener centralidades de nodos y aristas
    centr_nodos, centr_edges = self.obtener_centralidades_nodos_aristas()
  
    # tomar lsitas de nodos y aristas
    nodos = [v for v in self.G.nodes()]
    edges = [e for e in self.G.edges()]
  
    # hacer df
  
    # nodos
    df_centr_nodos = pd.DataFrame()
    df_centr_nodos["node"] = nodos
    df_centr_nodos["centralidad"] = centr_nodos
    df_centr_nodos = df_centr_nodos.sort_values("centralidad", ascending = False)
  
    # aristas
    df_centr_edges = pd.DataFrame()
    df_centr_edges["edges"] = edges
    df_centr_edges["centralidad"] = centr_edges
    df_centr_edges = df_centr_edges.sort_values("centralidad", ascending = False)
  
    return df_centr_nodos, df_centr_edges
 

  # ver si los tiempos reales de los agentes son los minimos
  def graficar_tiempos_agentes(self, alpha_hist = 0.6, alpha_scatter = 0.8, ax=None):
    '''
    Hace graficas para analizar como el tiempo real que toma a los agentes
    llega a su destino se desvia del tiempo minimo posible
    '''

    # tomar los agentes que si llegaron a su destino
    agentes_considerar = [coche for coche in self.historico_agentes.values()
                          if coche.tiempo_real < np.inf]

    # obtener los tiempos de los agentes
    tiempos_minimos = [agente.tiempo_minimo for agente in agentes_considerar]
    tiempos_reales = [agente.tiempo_real for agente in agentes_considerar]

    # si no hay ax hacer uno
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # histograma de los tiempos
    ax[0].hist(tiempos_minimos, bins=20, alpha=alpha_hist, label='Tiempo minimo')
    ax[0].hist(tiempos_reales, bins=20, alpha=alpha_hist, label='Tiempo real')
    ax[0].set_xlabel('Tiempo (steps)')
    ax[0].set_ylabel('Frecuencia')
    ax[0].set_title('Histograma de tiempos')
    ax[0].legend()

    # scatter plot de los tiempos
    ax[1].scatter(tiempos_minimos, tiempos_reales, alpha=alpha_scatter)
    ax[1].set_xlabel('Tiempo minimo (steps)')
    ax[1].set_ylabel('Tiempo real (steps)')
    ax[1].set_title('Scatter plot de tiempos')
    # trazar la linea y=x
    ax[1].plot([0, max(tiempos_minimos)], [0, max(tiempos_minimos)], color='red', linestyle='--', alpha = 0.4)

  # end graficar tiempo agentes


  def imprimir_estadisticas_tiempos_agentes(self):
    '''
    Imprime en pantalla las estadisticas de los tiempos de los agentes
    '''

    # solo considerar agentes que si llegaron a su destino
    agentes_modelos = [coche for coche in self.historico_agentes.values() if coche.tiempo_real < np.inf]

    # sacar valores
    tiempos_minimos = [agente.tiempo_minimo for agente in agentes_modelos]
    tiempos_reales = [agente.tiempo_real for agente in agentes_modelos]
    porcentaje_retraso = [100*agente.tiempo_real/agente.tiempo_minimo - 100 for agente in agentes_modelos if agente.tiempo_minimo != 0]
    veces_recalculan = [agente.num_recalculadas for agente in agentes_modelos]
    porcentaje_recalculadas_inutiles = [100*agente.num_recalculadas_inutiles/ agente.num_recalculadas for agente in agentes_modelos if agente.num_recalculadas != 0]

    # imprimir
    print(f"Informacion de los {len(agentes_modelos)} agentes que llegaron a su destino")
    print(f"Tiempo minimo promedio: {np.mean(tiempos_minimos):.3f}")
    print(f"Tiempo real promedio: {np.mean(tiempos_reales):.3f}")
    print(f"Porcentaje promedio de retraso: {np.mean(porcentaje_retraso):.3f}%")
    print(f"Promedio de veces que un agente recalcula: {np.mean(veces_recalculan):.3f}")
    if len(porcentaje_recalculadas_inutiles) > 0:
      print(f"Porcentaje promedio de recálculos inútiles: {np.mean(porcentaje_recalculadas_inutiles):.3f}%")



  # --------------------------------------------------------------------------

  # Atascos de todos los agentes del modelo

  # devuelve un df con la info de todos los atascos
  def get_atascos_agentes(self, min_espera = 10):
    '''
    Obtiene los atascos de de cada agente
    Un atasco se define como un nodo donde un agente
    estuvo parado mas de min_espera steps
    Devuelve un df con la informacion
    '''

    # hacer una lista de atascos
    atascos_agentes = []

    # agregar la de cada agente
    for agent in self.historico_agentes.values():
      # agregar sus atascos
      atascos_agentes += agent.reportar_atascos(min_espera = min_espera)

    # poner en un df
    df_atascos = pd.DataFrame(atascos_agentes)
    # ordenar segun tiempo
    if len(df_atascos) > 0:
      df_atascos = df_atascos.sort_values(by="tiempo", ascending = False)

    return df_atascos


  # ver los atascos
  def ver_atascos_agentes(self, min_espera = 10, ax=None):
    '''
    Ver los atascos de todos los agentes en el mapa
    Un atasco se define como un nodo donde un agente
    estuvo parado mas de min_espera steps
    '''

    # obtener los nodos que tienen atascos
    df_atascos = self.get_atascos_agentes(min_espera = min_espera)
    
    # si no hay atascos pues no hacer nada
    if df_atascos.shape[0] == 0:
        return 
    
    # tomar nodos con atascos
    nodos_atascos = df_atascos["nodo"].values

    # Definir colores y tamaños, segun si hay atasco o no
    colores_nodos = []
    tamaños_nodos = []

    # iterar en los nodos
    for nodo in self.G.nodes():
      # si presenta un atasco
      if nodo in nodos_atascos:
        colores_nodos.append(self.color_atasco)
        tamaños_nodos.append(self.node_size*5)
      # no presenta un atasco
      else:
        colores_nodos.append(self.color_vacio)
        tamaños_nodos.append(self.node_size)

    # Hacer la figura

    # si no se tiene ax hacer uno
    if ax is None:
      # hacer la figura
      fig, ax = plt.subplots(figsize=(8, 8))

    # dibujar en el ax
    nx.draw(self.G, self.pos, node_color = colores_nodos, edge_color = self.edge_color,
            node_size = tamaños_nodos, arrowsize = self.arrowsize, ax=ax)

  # --------------------------------------------------------------------------

  # Salvar informacion del modelo

  # funcion principal para salvar
  def salvar_modelo(self, folder_path = './'):
    '''
    Salvar informacion del modelo en la ruta respecificada
    
    '''

    # crear la carpeta si no existe
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    # Primero salvar la informacion del grafo

    # poner en un dict
    info_grafo = {
        "G": self.G_limpio, # guardar la version que no tiene info de agentes
        "pos": self.pos,
        "proba_origen": self.proba_origen,
        "proba_destino": self.proba_destino
    }
     # salvar
    with open(folder_path + "info_grafo.pkl", "wb") as f:
      pickle.dump(info_grafo, f)


    # Despues salvar info de las trayectorias de los agentes

    # por cada agente se salva info de la trayectoria que tomo
    dict_trayectorias = {} # iniciar vacio
    for agente_id, agente in self.historico_agentes.items(): # iterar todos los agentes
      # poner info de sus trayectorias
      dict_trayectorias[agente_id] = {
          "step_creation": agente.step_creation,
          "path": agente.path,
          "color": agente.color,
          }
    # ahora si salvar
    with open(folder_path  + 'info_paths.pkl', 'wb') as f:
      pickle.dump(dict_trayectorias, f)


    # Despues salvar la info de los bloqueos
    with open(folder_path  + 'info_bloqueos.pkl', 'wb') as f:
      pickle.dump(self.bloqueos, f)

    # Finalmente salvar la info del data collector

    # obtener df con info
    results_data_collector = self.datacollector.get_model_vars_dataframe()
    # salvar
    results_data_collector.to_csv(folder_path + "results_data_collector.csv")


  # end salvar_modelo

  # --------------------------------------------------------------------------

  # limpiar

def borrar_informacion(self):
    '''
    Borrar toda la información de la simulación para liberar memoria
    '''

    # eliminar información del grafo
    self.G = None
    self.pos = None
    self.proba_origen = None
    self.proba_destino = None

    # eliminar información de los agentes
    self.historico_agentes = {}

    # llamar al recolector de basura
    gc.collect()


