# -*- coding: utf-8 -*-
"""IA_IntroRL_ENSISA_2A_TD1_Prbl2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10AK6Hwfj3oA9L2xhq3P1E7fLtehnuN4g

> Import libraries to use
"""

import numpy as np

""">  # Introduction to numpy (Skip if you already are familiar)

>> Creating a 1D array
"""

a = np.array([1,2,3,4])
print(a)

""">> Creating a 2D array

"""

a = np.array([[1,2],[3,4]])
print(a)

""">> Creating an array full of zeros

"""

a = np.zeros(shape=(10))
print(a)
a = np.zeros(shape=(5,2))
print(a)

""">> Infinity in numpy"""

print(np.inf)

""">> Max and Argmax"""

a = np.array([2,1,4,3])
print(np.max(a))
print(np.argmax(a))

""">> From list to Numpy"""

l = [1,2,3,4]
print(l)
print(np.asarray(l))

""">> Random in numpy"""

# Array of Random integers ranging from 1 to 10 (with any size you want)
a = np.random.randint(low=1, high=10, size=(5,2))
print(a)

# Array of random elements of a list with any size you want
a = np.random.choice([0,1,2], size=(2,))

""">> Shapes in numpy"""

a = np.random.randint(low=1, high=5, size=(4,2))
print(a.shape)
print(a)

# Reshape a to a vector of shape = (8,1)
a = a.reshape((8,1))
print(a.shape)
print(a)

"""# Pre-defined utilities"""

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):

    pi_char = ['']

    for i in range(n):
        for j in range(n):

            if i == 0 and j == 0 or i == n-1 and j == n-1:

                continue

            pi_char.append(int_to_char[pi[i,j]])

    pi_char.append('')

    return np.asarray(pi_char).reshape(n,n)

"""# 1- Policy evaluation"""


def policy_evaluation(n, pi, v, Gamma, threshhold, max_iterations=1000):
    for _ in range(max_iterations):  # Limite le nombre d'itérations
        delta = 0
        for i in range(n):
            for j in range(n):
                # Si l'état est un état terminal (0,0 ou n-1,n-1), on ne fait rien
                if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                    continue

                # Calcul de la nouvelle valeur pour V[i,j]
                action = pi[i, j]
                dx, dy = policy_one_step_look_ahead[action]
                next_i, next_j = i + dx, j + dy

                # Si le mouvement est en dehors de la grille, l'agent reste dans sa position actuelle
                if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                    next_i, next_j = i, j

                # Calcul de la valeur attendue pour l'état (i, j)
                v_new = -1 + Gamma * v[next_i, next_j]
                delta = max(delta, abs(v_new - v[i, j]))
                v[i, j] = v_new

        # Condition d'arrêt
        if delta < threshhold:
            break

    return v

"""# 2- Policy improvement"""

def policy_improvement(n,pi,v,Gamma):
  """
    This function should return the new policy by acting in a greedy manner.
    The function should return as well a flag indicating if the output policy
    is the same as the input policy.

    Example:
      return new_pi, True if new_pi = pi for all states
      else return new_pi, False
  """

  pi_stable = True

  for i in range(n):
      for j in range(n):
          # Si l'état est terminal, on passe
          if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
              continue

          # Action actuelle
          old_action = pi[i, j]

          # Calcul des valeurs pour chaque action possible
          action_values = []
          for action in range(4):
              dx, dy = policy_one_step_look_ahead[action]
              next_i, next_j = i + dx, j + dy

              # Si le mouvement est en dehors de la grille, l'agent reste dans sa position actuelle
              if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                  next_i, next_j = i, j

              # Calcul de la récompense pour l'action
              action_value = -1 + Gamma * v[next_i, next_j]
              action_values.append(action_value)

          # Choisir l'action avec la valeur maximale (greedy)
          best_action = np.argmax(action_values)
          pi[i, j] = best_action

          # Vérification de stabilité de la politique
          if old_action != best_action:
              pi_stable = False

  return pi, pi_stable

"""# 3- Policy Initialization"""

def policy_initialization(n):
  """
    This function should return the initial random policy for all states.
  """
  pi = np.random.randint(low=0, high=4, size=(n,n))
  return pi

"""# 4- Policy Iteration algorithm"""

def policy_iteration(n,Gamma,threshhold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)

        if pi_stable:

            break

    return pi , v

"""# Main Code to Test"""

n = 4

Gamma = [0.8,0.9,1]

threshhold = 1e-4

for _gamma in Gamma:

    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)