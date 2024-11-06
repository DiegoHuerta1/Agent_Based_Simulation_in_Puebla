# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:21:41 2024

@author: diego
"""



import mesa
import numpy as np


'''
Clase del agente
'''


class Coche(mesa.Agent):
  '''
  Coche:
    - Aparece en un nodo del grafo
    - Intenta llegar a otro nodo 
    - Recalcula su ruta usando pesos en le grafo que es el entorno
  '''

  # constructor
  def __init__(self, unique_id, model, origen, destino, color, step_creation,
               tiempo_recalcular = 10):

    '''
    Para el constructor se especifica:
      EL ID del agente
      El modelo al que pertenece
      El nodo de origen
      El nodo de destino
      Color usando para visualizaciones
      El step donde el agente aparece en el modelo
      Cada cuantos steps va a recalcular el camino hacia su destino
    '''

    # constructor de la clase padre
    super().__init__(unique_id, model)

    # inicializar atributos
    self.unique_id = unique_id
    self.model = model
    self.origen = origen
    self.destino = destino
    self.color = color
    self.step_creation = step_creation  # step donde el agente aparece en el modelo
    self.esperando = False  # cuando el agente se crea pues obvio no esta esperando

    # poner al agente en el entorno y scheduler del modelo
    self.model.grid.place_agent(self, origen)
    self.model.schedule.add(self)

    # inicar atributos de camino
    self.node = origen     # localizacion de este agente
    self.path = [origen]   # su camino inicia en su origen
    self.next_steps = self.compute_path_to_destination()  # camino a seguir
    self.num_recalculadas = 0 # cuantas veces tiene que calcular el camino a seguir
    self.num_recalculadas_inutiles = 0 # cuantas veces se recalcula y es el mismo camino
    self.tiempo_recalcular = tiempo_recalcular # cada cuantos steps recalcular el camino
    self.steps_sin_recalcular = 0 # cuantos steps lleva siguiendo el mismo camino
    self.tiempo_real = np.inf # tiempo que se tarde en completar el recorrido (inf pues aun no termina)

  # end constructor

  # ----------------------------------------------------------
  # Calcular el camino hacia el objetivo

  # camino a seguir
  def compute_path_to_destination(self):
    '''
    Devuelve los nodos que se tienen que recorren,
    para llegar del nodo actual self.node
    al nodo destino self.destino
    No incluye al nodo actual en la lista, pues no se tiene que mover a este

    Para delimitar el camino al destino, se usa A* en el grafo de igraph
    Se utilizan los pesos en las aristas, obviamente se usan las de la ultima actualizacion
    '''

    # usar la funcion del modelo
    shortest_path = self.model.shortest_path_graph(self.node, self.destino)

    # quitar el primer nodo (el nodo actual)
    shortest_path = shortest_path[1:]

    return shortest_path


  # ------------------------------------------------------------

  # recalcular camino a seguir
  def recalcular_camino(self):
    '''
    Recalcular el camino a seguir para llegar al objetivo
    '''

    # calcular el camino
    nuevo_path = self.compute_path_to_destination()

    # si el nuevo path es igual al pasdo, esta recalculada es inutil
    if nuevo_path == self.next_steps :
      self.num_recalculadas_inutiles += 1

    # tomar el nnuevo camino
    self.next_steps = nuevo_path
    self.steps_sin_recalcular = 0
    self.num_recalculadas += 1


  # ------------------------------------------------------------

  # funcion para ver a donde se va a mover
  def seleccionar_nodo_movimiento(self):
    '''
    El agente escoje hacia donde se va a mover, o si se va a quedar quieto
    Se regresa el nodo donde se quiere mover (podria ser igual al actual)

    '''

    # si es que hay varios caminos a seguir (out degree mayor a 1), y se tiene que recalcular, hacerlo
    if self.model.G.out_degree(self.node) > 1 and self.steps_sin_recalcular >= self.tiempo_recalcular:
      self.recalcular_camino()
    # si aun no se tiene que recalcular
    else:
      # subir en uno esta cuenta
      self.steps_sin_recalcular += 1

    # tomar el paso que sigueen el camino hacia el destino
    nodo_siguiente = self.next_steps[0]

    # si esta vacio, mover a este
    if self.model.grid.is_cell_empty(nodo_siguiente):
      # quitar de los siguientes pasos
      self.next_steps.pop(0)
      return nodo_siguiente

    # si esta ocupado, quedarse quiero
    else:
      return self.node

  # end seleccionar_nodo_movimiento

  # ------------------------------------------------------------

  # funcion para mover al coche a un nodo
  def mover(self, nodo):
    '''
    Mueve al agente a un nodo especificado
    Funcion auxiliar de step()
    '''

    # primero ver si esta esperando
    # esta esperando si el nodo al que se va a mover es donde ya estaba
    if nodo == self.node:
      self.esperando = True
    else:
      self.esperando = False

    # mover
    self.model.grid.move_agent(self, nodo)
    # actualizar atributos
    self.node = nodo
    self.path.append(nodo)
  # end funcion mover


  # ------------------------------------------------------------
  ##########
  ## STEP ##
  ##########
  # funcion importante
  def step(self):
    '''
    Ejecuta un paso del agente.

    Selecciona un nodo a donde se va a mover
    Mueve al agente ahi
    '''

    # seleccionar a donde se va a mover
    nodo_mover = self.seleccionar_nodo_movimiento()

    # moverse ahi
    self.mover(nodo_mover)

  # end function step

  # ------------------------------------------------------------
  # ver si ya llego a su destino
  def maybe_terminar(self):
    '''
    Ver si este agente ya termino su ejecucion,
    esto es, ya llego a su destino
    '''
    # ver si esta en su destino
    if self.node == self.destino:
      # quitar del grafo y del scheduler
      self.model.grid.remove_agent(self)
      self.model.schedule.remove(self)
      # indicar cuando tiempo se tardo en hacer el recorrido
      self.tiempo_real = len(self.path) - 1
  # end maybe_terminar


  # reportar atascos, solo hacer cuando termina su ejecucion
  def reportar_atascos(self, min_espera = 10):
    '''
    Reporta los atascos que tuvo en su camino
    Un atasco es un nodo donde tuvo que esperar mas de min_espera steps
    Esta funcion debe de llamarse al final de su ejecucion
    '''
    # diccionario con pares  nodo:tiempo_parado_ahi
    dict_atascos_nodo = {}

    # ir guardando el nodo de potencial atasco
    nodo_ultimo = self.origen
    # ir viendo cuanto tiempo lleva parado
    tiempo_parado = 0

    # iterar en el camino del nodo (no iniciar en el origen)
    for nodo in self.path[1:]:

      # si este nodo es igual al anterior
      if nodo == nodo_ultimo:
        # aumentar el tiempo parado
        tiempo_parado += 1

        # si ya se considera atascos
        if tiempo_parado >= min_espera:
          # poner en el diccionario
          dict_atascos_nodo[nodo_ultimo] =  tiempo_parado
        # end if- no se considera atascos

      # end if - ver si se esta en el mismo nodo que antes

      # si no se esta en el mismo nodo que antes
      else:

        # se resetean las variables
        nodo_ultimo = nodo
        tiempo_parado = 0
      # end else
    # end for - iterar en el camino

    # los atascos se tienen en un diccionario de pares
    # nodo: tiempo atascado ahi
    # convertir a una lista de diccionarios
    # donde cada diccionario es de la forma {"nodo": nodo, "tiempo": tiempo, "agente": id}

    # inicar lista vaica
    atascos_nodo = []

    # iterar en lo que se tienee
    for nodo, tiempo in dict_atascos_nodo.items():
      # agregar a la lista
      atascos_nodo.append({"nodo": nodo, "tiempo": tiempo, "agente": self.unique_id})

    return atascos_nodo