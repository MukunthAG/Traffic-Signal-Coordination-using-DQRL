# Traffic-Signal-Coordination-using-DQRL

**THIS IS AN EXPERIMENTAL VERSION!!** 

**Traffic Model**
1. All Vehicles obey the traffic rules
2. Vehicles can accelerate and decelerate depending upon the need
3. Each vehicle entering the network has an assinged path to its destination (may not be the shortest)
4. The number of vehicles entering the network is sampled from a binomial distribution with its mean as the expected arrival rate (an important parameter, which represents the amount of traffic)
5. Three signals in series is considered

**Methods Tested**

Here's a summary of all the significant methods (see branches), tested for a 15 minute _rush hour_ (vehicles will be continuously entering only for 15 minutes) on three different arrival rates.

We can observe that centralised DQRL performs well for all three arrival rates. Eventhough multi-agent DQRL outperforms centralised RL by a slight margin, it doesn't converge well most of the times. 

<img src="https://user-images.githubusercontent.com/72994221/163859598-aa6cda69-4f70-43ac-b45e-a0942853f966.png" width="600">

**Testing Methodology**

Given the traffic model, and the traffic light control logic (based on pressure at each lane), the **episode duration till which all the vehicles leave the intersection** is taken as a sole criteria, this may not be practical (since vehicles enter the network continuously), but for the testing purposes, this is a perfectly valid criteria to assess the algorithm's perfomance.

![ezgif-4-69d838eba9](https://user-images.githubusercontent.com/72994221/163852579-273333ce-f8a2-4359-a94c-3553ca69ee75.gif)

**Conclusion**

DQRL algorithms significantly outperforms the fixed time control under experimental setting. Although since the model is not so realistic, theres a lot of room for improvement.
