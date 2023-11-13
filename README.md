# parallel_job_scheduling

## Input file should be formatted as follows :
### The first line contains the number of computer centres. 
### For each computer centre,
 – computer centre number, number of facility rooms P
 – The next line contain P space-separated integers i.e. facility room numbers
 – The next line contain P space-separated integers i.e. capacity C of each facility room
### The next line contain the total number of requests R.
### The next R lines contains details of each request from user which are followed below,
– Request ID, computer centre number, facility room number, Starting slot number, number of slots to be reserved.

## Optimizations used :
 - High degree of parallelization given (almost constant runtime with respect to number of requests.)
 - Graph created in CSR format corresponding to the cluster's architecture.
 - Matching algorithm applied.
