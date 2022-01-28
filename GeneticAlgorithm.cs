using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra; //matrices


public class GeneticAlgorithm : MonoBehaviour
{
    [Header("Refernce")]
    public CarController controller;

    [Header("Population")]
    public int initialPopulation = 300;

    [Range(0.0f, 1.0f)]
    public float mutationRate = 0.04f;

    [Header("Crossover Control")]
    public int selectBest = 146;
    public int selectWorst = 4;
    public int crossoverAmount = 4;

    private List<int> genePool = new List<int>();

    private int selectedCount;
    private NeuralNetwork[] population;

    [Header("Population tracker")]
    public int currentGen = 0;
    public int currentChrom = 0;

    private void Start()
    {
        InitialisePopulation();
    }

    private void InitialisePopulation()
    {
        population = new NeuralNetwork[initialPopulation];
        RandomPopulate(population, 0);
        ResetToCurrentGenome();
    }

    private void RandomPopulate(NeuralNetwork[] newPop, int i)
    {
        while (i < initialPopulation)
        {
            newPop[i] = new NeuralNetwork();
            newPop[i].Initialise(controller.layers, controller.neurons);
            i++;
        }
    }

    private void ResetToCurrentGenome()
    {
        controller.ResetNetwork(population[currentChrom]);
    }

    //car controller sends fitness value and network details for evaluation on death
    public void Death(float passedFitness, NeuralNetwork nn)
    {
        if (currentChrom < population.Length - 1)
        {
            population[currentChrom].fitness = passedFitness;
            currentChrom++;
            ResetToCurrentGenome();
        }
        else
        {
            Repopulate();
        }
    }

    private void Repopulate()
    {
        controller.decreaseEpoch();
        controller.checkEpoch();
        genePool.Clear();
        currentGen++;
        selectedCount = 0;
        SortPopulation();
        NeuralNetwork[] newPop = PickBest();

        Crossover(newPop);
        Mutation(newPop);

        RandomPopulate(newPop, selectedCount);

        currentChrom = 0;
        ResetToCurrentGenome();
    }

    public void Crossover(NeuralNetwork[] newPop)
    {
        for (int i = 0; i < crossoverAmount; i += 2)
        {
            int point1 = i;
            int point2 = i + 1;
            if (genePool.Count > 0)
            {
                for (int j = 0; j < 100; j++)
                {
                    point1 = genePool[Random.Range(0, genePool.Count)];
                    point2 = genePool[Random.Range(0, genePool.Count)];

                    if (point1 != point2)
                    {
                        break;
                    }
                }
            }

            NeuralNetwork offspring1 = new NeuralNetwork();
            NeuralNetwork offspring2 = new NeuralNetwork();
            offspring1.Initialise(controller.layers, controller.neurons);
            offspring2.Initialise(controller.layers, controller.neurons);
            offspring1.fitness = 0;
            offspring2.fitness = 0;

            for (int k = 0; k < offspring1.weights.Count; k++)
            {
                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    offspring1.weights[k] = population[point1].weights[k];
                    offspring2.weights[k] = population[point2].weights[k];
                }
                else
                {
                    offspring1.weights[k] = population[point2].weights[k];
                    offspring2.weights[k] = population[point1].weights[k];
                }
            }

            for (int l = 0; l < offspring1.biases.Count; l++)
            {
                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    offspring1.biases[l] = population[point1].biases[l];
                    offspring2.biases[l] = population[point2].biases[l];
                }
                else
                {
                    offspring1.biases[l] = population[point2].biases[l];
                    offspring2.biases[l] = population[point1].biases[l];
                }
            }

            newPop[selectedCount] = offspring1;
            selectedCount++;
            newPop[selectedCount] = offspring2;
            selectedCount++;
        }
    }

    public void Mutation(NeuralNetwork[] newPop)
    {
        for (int i = 0; i < selectedCount; i++)
        {
            for (int j = 0; j < newPop[i].weights.Count; j++)
            {
                if (Random.Range(0.0f, 1.0f) < mutationRate)
                {
                    newPop[i].weights[j] = MutateMatrix(newPop[i].weights[j]);
                }
            }
        }
    }
    Matrix<float> MutateMatrix(Matrix<float> A)
    {
        int randomPoints = Random.Range(1, (A.RowCount * A.ColumnCount) / 7);
        Matrix<float> B = A;

        for (int i = 0; i < randomPoints; i++)
        {
            int randomColumn = Random.Range(0, B.ColumnCount);
            int randomRow = Random.Range(0, B.RowCount);
            B[randomRow, randomColumn] = Mathf.Clamp(B[randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return B;
    }

    private void SortPopulation()
    {
        for (int i = 0; i < population.Length; i++)
        {
            for (int j = 0; j < population.Length; j++)
            {
                if (population[i].fitness < population[j].fitness)
                {
                    NeuralNetwork temp = population[i];
                    population[i] = population[j];
                    population[j] = temp;
                }
            }
        }
    }

    private NeuralNetwork[] PickBest()
    {
        NeuralNetwork[] newPop = new NeuralNetwork[initialPopulation];

        for (int i = 0; i < selectBest; i++)
        {
            newPop[selectedCount] = population[i].InitialiseCopy(controller.layers, controller.neurons);
            newPop[selectedCount].fitness = 0;
            selectedCount++;

            int f = Mathf.RoundToInt(population[i].fitness * 10);
            for (int x = 0; x < f; x++)
            {
                genePool.Add(i);
            }
        }

        for (int j = 0; j < selectWorst; j++)
        {
            int last = population.Length - 1;
            last -= j;

            int g = Mathf.RoundToInt(population[last].fitness * 10);
            for (int y = 0; y < g; y++)
            {
                genePool.Add(last);
            }
        }
        return newPop;
    }

    public int getChromCount()
    {
        return currentChrom;
    }

    public int getGenCount()
    {
        return currentGen;
    }
}
