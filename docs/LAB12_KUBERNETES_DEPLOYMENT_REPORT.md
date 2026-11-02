\# LAB 12 – Kubernetes Deployment Report

\## COVID-19 MLOps Project



\### 1. Objective

To deploy the containerized COVID-19 prediction API on a local Kubernetes cluster using Docker Desktop Kubernetes.



\### 2. Kubernetes Environment

\- Kubernetes Context: `docker-desktop`

\- Kubernetes Version: `v1.36.1`

\- Node: `desktop-control-plane`

\- Node Status: Ready

\- Docker Image: `covid-mlops-api:latest`



\### 3. Deployment Configuration

Created `k8s/deployment.yaml` to deploy the COVID-19 API.



\- Deployment Name: `covid-api-deployment`

\- Container Name: `covid-api`

\- Container Port: `8000`

\- Initial Replicas: 2

\- Final Replicas: 3



\### 4. Service Configuration

Created `k8s/service.yaml`.



\- Service Name: `covid-api-service`

\- Service Type: ClusterIP

\- Service Port: 8000

\- Target Port: 8000



\### 5. API Testing

Used Kubernetes port forwarding:



`kubectl port-forward service/covid-api-service 8004:8000`



Accessed Swagger UI at:



`http://127.0.0.1:8004/docs`



Verified that the API executed successfully through Swagger UI.



\### 6. Horizontal Scaling

Scaled the deployment from 2 to 3 replicas using:



`kubectl scale deployment covid-api-deployment --replicas=3`



Verified that all three pods reached Running and Ready status.



\### 7. Self-Healing Test

Deleted one running COVID API pod manually.



Kubernetes automatically created a replacement pod through the Deployment controller.



Verified that the deployment returned to 3/3 Ready replicas.



\### 8. Service Endpoint Verification

Verified using:



`kubectl get endpoints covid-api-service`



Three pod IP addresses were registered on port 8000.



\### 9. Final Result

\- Kubernetes deployment successful.

\- ClusterIP service created successfully.

\- Swagger API testing successful.

\- Scaling to three replicas successful.

\- Automatic pod recovery verified.

\- Three service endpoints available.



\### 10. Conclusion

The COVID-19 MLOps API was successfully deployed on a local Kubernetes cluster. Scaling and self-healing capabilities were tested and verified, demonstrating successful container orchestration using Kubernetes.



\*\*Status: LAB 12 COMPLETED\*\*

