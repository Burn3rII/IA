import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update the Q table using the Q-learning formula:
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
    """
    # Calculer la valeur de la mise à jour de Q(s, a)
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilon):
    """
    Choisit une action en suivant l'epsilon greedy policy.
    Avec probabilité epsilon, choisit une action aléatoire (exploration),
    sinon choisit l'action ayant la valeur Q(s, a) la plus élevée (exploitation).
    """
    # Choisir une action de manière aléatoire avec probabilité epsilon (exploration)
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    # Sinon, choisir l'action ayant la valeur Q la plus élevée
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")

    # Initialiser la Q-table avec des zéros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Définir les hyperparamètres
    alpha = 0.1  # Taux d'apprentissage
    gamma = 0.9  # Facteur de discount
    epsilon = 0.2  # Paramètre epsilon pour l'exploration
    n_epochs = 1000  # Nombre d'épisodes d'entraînement
    max_itr_per_epoch = 1000  # Nombre max d'itérations par épisode
    rewards = []

    for e in range(n_epochs):
        # Réinitialiser l'environnement au début de chaque épisode
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            # Sélectionner une action en suivant la politique epsilon-greedy
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            # Effectuer l'action et obtenir le prochain état et la récompense
            Sprime, R, done, _, info = env.step(A)

            # Mettre à jour la récompense totale de l'épisode
            r += R

            # Mettre à jour la Q-table
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha,
                               gamma=gamma)

            # Passer à l'état suivant
            S = Sprime

            # Arrêter si l'épisode est terminé (passager déposé à la destination)
            if done:
                break

        print("Episode #", e + 1, " - Reward: ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # Afficher l'évolution des récompenses
    plt.plot(range(n_epochs), rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.title("Training Performance over Episodes")
    plt.show()

    print("Training finished.\n")

    # Fermer l'environnement
    env.close()
