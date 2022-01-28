using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System;

[RequireComponent(typeof(NeuralNetwork))]
public class CarController : MonoBehaviour
{
    private Vector3 startPos, startRot;
    private NeuralNetwork network;
    private Vector3 moveInput;

    public GeneticAlgorithm geneticAlgorithm;

    [Range(-1f,1f)]
    public float a,t;

    public float timeSinceStart = 0f;

    [Header("Fitness values")]
    public float overallFitness;
    public float distanceMultipler = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 0.1f;

    [Header("Hyperparameters")]
    public int layers = 1;
    public int neurons = 24;
    public int epoch = 200;

    private Vector3 lastPos;
    private float totalDist;
    private float avgSpeed;

    [Header("Results")]
    public int generationCount = 0;
    public int solutionCount = 0;

    private float aSensor,bSensor,cSensor;
    private List<int> generationList = new List<int>();
    private List<int> solutionList = new List<int>();
    private string filePath;

    private void Awake()
    {
        geneticAlgorithm = GameObject.Find("_GA").GetComponent<GeneticAlgorithm>();
        startPos = transform.position;
        startRot = transform.eulerAngles;
        network = GetComponent<NeuralNetwork>();
    }

    public void ResetNetwork(NeuralNetwork nn)
    {
        network = nn;
        Reset();
    }
    

    public void Reset()
    {
        timeSinceStart = 0f;
        totalDist = 0f;
        avgSpeed = 0f;
        lastPos = startPos;
        overallFitness = 0f;
        transform.position = startPos;
        transform.eulerAngles = startRot;
    }

    private void OnCollisionEnter(Collision collision)
    {
        Death();
    }

    private void FixedUpdate()
    {
        InputSensors();
        lastPos = transform.position;

        (a, t) = network.StartNetwork(aSensor, bSensor, cSensor);

        MoveCar(a,t);

        timeSinceStart += Time.deltaTime;
        CalculateFitness();
    }

    private void Death()
    {
        GameObject.FindObjectOfType<GeneticAlgorithm>().Death(overallFitness, network);
    }

    private void CalculateFitness()
    {
        totalDist += Vector3.Distance(transform.position,lastPos);
        avgSpeed = totalDist/timeSinceStart;

       overallFitness = (totalDist*distanceMultipler)+(avgSpeed*avgSpeedMultiplier)+(((aSensor+bSensor+cSensor)/3)*sensorMultiplier);

        if ((timeSinceStart > 25 && overallFitness < 50) || overallFitness > 1350)
        {
            if(overallFitness > 1350)
            {
                generationCount = geneticAlgorithm.getGenCount() + 1;
                solutionCount++;
                generationList.Add(generationCount);
                solutionList.Add(solutionCount);
            }
            Death();
        }
    }

    private void InputSensors()
    {
        Vector3 right = (transform.forward+transform.right);
        Vector3 centre = (transform.forward);
        Vector3 left = (transform.forward-transform.right);

        Ray vision = new Ray(transform.position, right);
        RaycastHit hit;

        if (Physics.Raycast(vision, out hit))
        {
            aSensor = hit.distance/20;
            Debug.DrawLine(vision.origin, hit.point, Color.red);
        }

        vision.direction = centre;

        if (Physics.Raycast(vision, out hit))
        {
            bSensor = hit.distance/20;
            Debug.DrawLine(vision.origin, hit.point, Color.red);
        }

        vision.direction = left;

        if (Physics.Raycast(vision, out hit))
        {
            cSensor = hit.distance/20;
            Debug.DrawLine(vision.origin, hit.point, Color.red);
        }

    }

    public void MoveCar (float v, float h)
    {
        moveInput = Vector3.Lerp(Vector3.zero,new Vector3(0,0,v*11.4f),0.02f);
        moveInput = transform.TransformDirection(moveInput);
        transform.position += moveInput;

        transform.eulerAngles += new Vector3(0, (h*90)*0.02f,0);
    }

    public void decreaseEpoch()
    {
        epoch--;
    }

    public void checkEpoch()
    {
        if(this.epoch == 0)
        {
            writeToFile();
            Time.timeScale = 0;
        }
    }

    private void writeToFile()
    {
        float ratio = 0; ;
        filePath = getPath();
        StreamWriter writer = new StreamWriter(filePath);
        writer.WriteLine("Solution Count,Generation Count,Ratio");

        for(int i = 0; i < Math.Max(solutionList.Count, generationList.Count); i++)
        {
            if(i < solutionList.Count)
            {
                writer.Write(solutionList[i]);
            }
            writer.Write(",");

            if(i < generationList.Count)
            {
                writer.Write(generationList[i]);
            }
            writer.Write(",");
            ratio = (float)solutionList[i] / (float)generationList[i];
            writer.Write(ratio);
            writer.Write(System.Environment.NewLine);
        }

        writer.Flush();
        writer.Close();
    }

    private string getPath()
    {
        return Application.dataPath + "sols.csv";
    }
}
