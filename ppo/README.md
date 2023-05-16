#### PPO `2017`

---

- `TRPO`
    
        지금 얻은 데이터로 가능한 큰 step 만큼 업데이트하고 싶은데..
        그렇다고 너무 멀리가면 성능이 떨어질 수도 있고... 어떡하지?

        trpo : 이 문제를 복잡한 second-order method로 풀었다. (2차 미분을 구해서 ...)
        ppo  : first-order method !


---

- `PPO`

        트릭을 사용해서 새로운 policy를 기존의 policy와 가깝도록 유지하게 해준다.
        구현이 매우매우 간단하고 practically 좋은 성능을 보여준다. 
 
        

