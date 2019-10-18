# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Mingyan Zhao>
<Math 321>
<10/30/2018>
"""
import numpy as np
from collections import deque
import networkx as nx
from matplotlib import pyplot as plt


# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d.keys():
            self.d[n] = set()
        else:
            pass

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        #add nodes
        self.add_node(u)
        self.add_node(v)
        #add edges
        self.d[u].add(v)
        self.d[v].add(u)


    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        #check if it is in the graph
        if n not in self.d.keys():
            raise KeyError("The node " + str(n) + " is not in the graph.")
        #remove all the edges that link to to n
        for i in self.d[n]:
            self.d[i].remove(n)
        self.d.pop(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        #check if u,v are in the graph
        if u not in self.d.keys():
            raise KeyError("there is no edge between the nodes.")
        if v not in self.d.keys():
            raise KeyError("there is no edge between the nodes.")
        #remove edges
        self.d[u].remove(v)
        self.d[v].remove(u)

    # Problem 2
    def traverse(self, start):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # visits all the nodes of a graph (connected component) using BFS

        # keep track of all visited nodes
        explored = []
        # keep track of nodes to be checked
        queue = [start]

        # keep looping until there are nodes still to be checked
        while queue:
            # pop shallowest node (first node) from queue
            node = queue.pop(0)
            if node not in explored:
                # add node to list of checked nodes
                explored.append(node)
                neighbours = self.d[node]

                # add neighbours of node to queue
                for neighbour in neighbours:
                    queue.append(neighbour)
        return explored

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """

        if source not in self.d.keys():
            raise KeyError("The node " + str(source) + " is not in the graph.")
        if target not in self.d.keys():
            raise KeyError("The node " + str(target) + " is not in the graph.")

        # keep track of explored nodes
        explored = []
        # keep track of all the paths to be checked
        queue = [[source]]

        # return path if start is goal
        if source == target:
            return [source]

        # keeps looping until all possible paths have been checked
        while queue:
            # pop the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            if node not in explored:
                neighbours = self.d[node]
                # go through all neighbour nodes, construct a new path and
                # push it into the queue
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    # return path if neighbour is goal
                    if neighbour == target:
                        return new_path

                # mark node as explored
                explored.append(node)

        # in case there's no path between the 2 nodes
        raise KeyError("a connecting path doesn't exist.")


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.movie_titles = set()
        self.actor_names = set()
        self.Graph = nx.Graph()

        #store each line
        with open(filename, 'r') as myfile:
            contents = myfile.readlines()
        #store the title and actors
        for i in range(len(contents)):
            lines = []
            lines = str(contents[i]).strip().split("/")
            self.movie_titles.add(lines[0])
            for j in range(1, len(lines)):
                self.actor_names.add(lines[j])
                self.Graph.add_edge(lines[0],lines[j])

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints.
            (int): the number of steps from source to target, excluding movies.
        """
        #return the shortest_path and length
        return nx.shortest_path(self.Graph, source, target), nx.shortest_path_length(self.Graph, source, target)//2
    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #store all the path from actors to the target
        list = []
        x = nx.shortest_path_length(self.Graph, target)
        #only save the lengths for actors
        for i in x.keys():
            if i in self.actor_names:
                list.append(x[i])
                
        list = np.divide(list,2)
        #graph
        domain = len(list)
        plt.hist(list,bins=[i-.5 for i in range(8)])
        plt.xlabel("actors")
        plt.ylabel("path length")
        plt.title("length from actor to target")
        plt.show()
        return np.average(list)
