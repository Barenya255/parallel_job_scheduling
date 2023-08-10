/*
 * The following code is for assignment 4. Modeling the problem as a tripartite graph and storing a log request as an csr list reduces the space complexity
 * of the probem. 
 * 
 * Any other storage solution will result in either overflowing the DRAM memory or an allignment error. ( GPUs have a hard requirement for contiguous memory allocation )
 * Also other storage solutions maybe used if sorting the entire request array in CPU itself
 * 
 * The vertex space consists of center nodes, facility nodes and request nodes.
 * 
 * There are edges between center nodes and facility nodes and facility nodes and request nodes.
 * 
 * Now the capacity of each facility is the capacity of the edge going from that facility to it's center.

 * It is a lot like a max Flow problem
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here
__global__ void printArr (int *arr, int size) {
    /**
     * A debugging kernel. Primarily, it prints out the arr inside device through a kernel call and therefore, through anywhere.
    */


    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }printf("\n");
}


__global__ void exPreSum(int *d_arr, int *result, int N) {
    /**
     *  A kernel to calculate prefix sum for the kernel.
     *  It takes in the array, on which to perform prefix sum.
     *  It takes in an auxiliary array on to which must store the last elements.
     *  Future versions to be more flexible, probably via an iteration.
     *  This is for beta testing only.
    */


   // allocate shared memory.
    __shared__ int temp[1024];
    

    // compute global index and check for any out of bounds.. Fill with zeros otherwise.
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId < N) {
        temp[threadIdx.x] = d_arr[globId];
    } else {
        temp[threadIdx.x] = 0;
    }
    
    
    __syncthreads();
    
    int tmp;
    
    
    // Naive implementation of the prefix sum that uses two syncthreads instead of one.
    for (int offset = 1; offset < N; offset*=2) {
        if (threadIdx.x >= offset) {
            tmp = temp[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            atomicAdd(&temp[threadIdx.x], tmp);
        }
        __syncthreads();
    }
    

    // copy back what is needed.
    if(globId < N)
        d_arr[globId] = temp[threadIdx.x];
    __syncthreads();
    

    // to deal with inconsistencies accross blocks
    if (result != NULL and threadIdx.x == 1023) {
        //printf(" must add %d to result[%d]\n", blockIdx.x, temp[1023]);
        result[blockIdx.x] = temp[blockDim.x - 1];
    }
}


__global__ void addUp (int *d_arr, int * result, int N) {
    /* Primer function for the prefix sum over multiple blocks.*/
    
    
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= N) return;
    
    
    int adder = 0;
    if (blockIdx.x > 0) {
        adder = result[blockIdx.x-1];
    }
    
    
    //printf("adding to %d and adder value is: %d\n", globId, adder);
    d_arr[globId] += adder;
    
}


__global__ void copy (int *destination, int *source, int N) {
    /* The copy kernel will basically copy elements of destination to source within the device.*/
    
    
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId+1 >= N) return;
    if (globId == 0) {
        destination[globId] = 0;
    }
    
    destination[globId+1] = source[globId];
}


__host__ void preSum (int *d_arr, int *d_count, int N) {
    /* The parent function used to calculate prefix sum. Call this from the main code to see results.*/
    
    
    // Upto three levels possible and gives accurate values for N = INT_MAX/2. Should be enough for this problem.
    int numBlocks = ceil(float(N) / 1024);
    copy<<<numBlocks, 1024>>> (d_arr, d_count, N);
    cudaDeviceSynchronize();

    int* d_result, *d_result1;
    cudaMalloc(&d_result, (numBlocks) * sizeof(int));
    cudaMemset(d_result, 0, (numBlocks) * sizeof(int));
    cudaDeviceSynchronize();
    int numBlocks2 = ceil(float(numBlocks) / 1024);
    
    
    cudaMalloc(&d_result1, (numBlocks2) * sizeof(int));
    cudaMemset(d_result1, 0, (numBlocks2) * sizeof(int));
    
    cudaDeviceSynchronize();
    
    exPreSum<<<numBlocks, 1024>>>(d_arr, d_result, N);
    
    if (numBlocks > 1){
        exPreSum<<<numBlocks2, 1024>>>(d_result, d_result1, numBlocks);

        int numBlocks3 = ceil(float(numBlocks2) / 1024); 
        if (numBlocks2 > 1){
            exPreSum<<<numBlocks3, 1024>>> (d_result1, NULL, numBlocks2);
            addUp<<<numBlocks2, 1024>>> (d_result, d_result1, numBlocks);
        }

        addUp<<<numBlocks, 1024>>>(d_arr, d_result, N);
    }
      
    cudaFree(d_result);
    cudaFree(d_result1);
    
}


__global__ void fixCapacity (int *fixedCap, int *d_cap, int size) {
    /**
     *  This kernel is used to remove all padded zeros from the end of the capacity array.
     *  This is instrumental in the craetion of the csr list for capacity.
    */


    // calculate globally unique id of each thread and return if out of bounds.
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= size) return;
    
    
    // copy to the new array.
    fixedCap[globId] = d_cap[globId];
}


__global__ void fixFac (int *fixed_facs, int *facs, int size) {
    /**
     *  This kernel is used to remove all padded zeros from the end of the array containing facIds.
     *  This again is for implementing a csr List for the facs.
    */


    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= size) return;
    
    
    fixed_facs[globId] = facs[globId];
}


__global__ void generateOffset (int *f_offset, int *d_offset, int *req_cen, int *req_fac, int size) {
    /**
     * This kernel is used to basically calculate the offset for the csrList of edges going from the facility set to the request set.
     * This particular kernel does half of the total work.
     * It just calcultes count which is the number of requests in each unique facility.
    */


    // generate a globally unique id and return if out of bounds.
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= size) return;
    
    
    // calculate required indices.
    int cen = req_cen[globId];
    int fac = req_fac[globId];
    int index = d_offset[req_cen[globId]] + req_fac[globId];

    
    // atomic Add the offsets.
    atomicAdd(&f_offset[index], 1);
}


__global__ void prefixOffset (int *f_offset, int *d_count, int k1) {
    /**
     * This is to calculate the prefix sum of the offset array created by the previous kernel.
     * This version is to be deprecated in the future when a reductive implementation will take over.
     * An inclusive prefix sum is to be calculated.
    */

    // prefix sum the offset array.
    for (int i = 1; i < k1; i++) {
        f_offset[i] += f_offset[i-1] + d_count[i-1];
    }
    
    f_offset[0] = 0;
}


__global__ void pushRequests (int *d_offset, int *f_offset, int *f_csr, volatile int *helper, int *req_cen, int *req_fac, int R) {
    /**
     * This kernel basically pushes all the requests into the csr list indices corresponding to their particular facility rooms (unique).
     * There is no sorting or ordering done here.
     * Future version to be different and updates to be made available.
    */

    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= R) return;
    
    
    // calculate required indices for the requested facility.
    int cen_no = req_cen[globId];
    int fac_no = req_fac[globId];

    
    // calculate the facility offset for the requested facility.
    int fac_offset = d_offset[cen_no]+fac_no;
    int csrIndex = f_offset[fac_offset];

    
    // update index and get the current value for the csr List of the requested facility.
    int index = atomicAdd((int*)&helper[fac_offset], 1);
    f_csr[index+csrIndex] = globId;
}


__global__ void sortRequests (int *d_count, int *f_offset, int *f_csr, int *req_cen, int *req_fac, int k1) {
    /**
     * This sorts the request numbers allocated to the particular facility number (unique).
     * This avoids sorting the entire array and it sorts only small segments in parallel.
    */


    // declare the uuid for the thread and return if out of bounds.
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= k1) return;


    // calculate index into the csr list.
    int csrIndex = f_offset[globId];
    int N = d_count[globId];
    

    // sort the requests for this particular facility number. This results in multiple sorts being done in parallel.
    // Also not many elements amortized will belong to same facility number. Therefore time complexity improves.
    for (int i = csrIndex; i < csrIndex+N; i++) {
        for (int j = i+1; j < csrIndex+N; j++){
            if (f_csr[i] > f_csr[j]) {
                int temp = f_csr[i];
                f_csr[i] = f_csr[j];
                f_csr[j] = temp;
            }
        }
    }
}


__global__ void manualMemset (int *capacity, int ele) {
    /**
     * As cudaMemset can only set memory per byte, it is worthless when memsetting to actual integers (other than 0 and -1)
     * This requires the presence of dynamic parallelism.
    */


    capacity[threadIdx.x] = ele;
}


__global__ void scheduleRequests (int *d_offset, int *csrCAP, int *d_count, int *f_offset, int *f_csr, int *req_cen, int *req_fac, int *req_start, int *req_slots, int * succ_reqs, int k1) {
    /**
     * Assign requests and schedule whatever requests can be scheduled. 
     * To be merged with pushRequests in the near future.
    */

    
    // calculate the uuid of the thread running in the grid and proceed only if within bounds of requirement.
    int globId = blockIdx.x*blockDim.x+threadIdx.x;
    if (globId >= k1) return;
    
    
    // calculate the offset for accessing the facility csr list.
    int csrIndex = f_offset[globId];
    int N = d_count[globId];
    
    
    // create and populate capacities for all 24 hours of the day.
    int cap = csrCAP[globId];
    int* capacity = (int*) malloc (sizeof(int)*24);
    manualMemset<<<1,24>>>(capacity, cap);
    cudaDeviceSynchronize();
    

    // flag will decide whether to commit or not.
    int flag = 0;
    
    
    for (int i = csrIndex; i < csrIndex+N; i++) {
        int r_no = f_csr[i];
        int r_start = req_start[r_no]-1;
        int r_slots = req_slots[r_no];
        int r_end = r_start+r_slots;
        int cen = req_cen[r_no];
        
        for (int j = r_start; j < r_end; j++) {
            atomicAdd(&capacity[j], -1);
            if (capacity[j] < 0) flag = 1;
        }
        
        if (flag == 1) {
            // roll back all changes. Commit Failed.


            for (int j = r_start; j < r_end; j++)
                atomicAdd(&capacity[j], 1);
        } else {
            // commit acknowledged. Save changes, update number of successful requests for the particular center.


            atomicAdd (&succ_reqs[cen], 1);
        }
        flag = 0;
    }


    // Free up the array as it is done and no longer needed.
    cudaFree(capacity);
}
//***********************************************


int main(int argc, char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
    
    
    int* unified_offset = (int*)malloc(N*sizeof(int));
    int preFacSum = 0;
    for (int i = 0; i < N; i++){
        unified_offset[i] = preFacSum;
        preFacSum += facility[i];
    }
    
    
    // intialize, allocate, and copy for request centers and request numbers.
    int *d_req_cen, *d_req_fac;
    cudaMalloc (&d_req_cen, sizeof(int)*R);
    cudaMalloc (&d_req_fac, sizeof(int)*R);
    cudaMemcpy (d_req_cen, req_cen, sizeof(int)*R, cudaMemcpyHostToDevice);
    cudaMemcpy (d_req_fac, req_fac, sizeof(int)*R, cudaMemcpyHostToDevice);
    
    
    // intialize, allocate, and copy for request start poitns and request slot numbers.
    int *d_req_start, *d_req_slots;
    cudaMalloc (&d_req_start, sizeof(int)*R);
    cudaMalloc (&d_req_slots, sizeof(int)*R);
    cudaMemcpy (d_req_start, req_start, sizeof(int)*R, cudaMemcpyHostToDevice);
    cudaMemcpy (d_req_slots, req_slots, sizeof(int)*R, cudaMemcpyHostToDevice);
    
    
    // This d_offset is the offset array for the computer centers. 
    // Can be used to index into capacity csr list or fac id csr list.
    int *d_offset;
    cudaMalloc (&d_offset, sizeof(int)*N);
    cudaMemcpy (d_offset, unified_offset, sizeof(int)*N, cudaMemcpyHostToDevice);
    
    
    // d_count to store bucket sizes for each array.
    int *d_count;
    cudaMalloc (&d_count, sizeof(int)*k1);
    
    
    // f_offset to store the indexes into the f_csrList which will contain the request numbers.
    int *f_offset;
    cudaMalloc (&f_offset, sizeof(int)*k1);
    
    
    // f_csr contains request numbers and with appropriate indices, we can know which unique facility room it belongs to.
    int *f_csr;
    cudaMalloc (&f_csr, sizeof(int)*R);
    
    
    // helper array used to push elements into the f_csr list.
    volatile int *d_helper;
    cudaMalloc (&d_helper, sizeof(int)*k1);
    cudaMemset ((int*)d_helper, 0, sizeof(int)*k1);
    

    // d_cap and fixed_cap were created to fix the appended zeros in the capacity array.
    int *d_cap, *fixed_cap;
    cudaMalloc (&d_cap, sizeof(int)*N*max_P);
    cudaMemcpy (d_cap, capacity, sizeof(int)*N*max_P, cudaMemcpyHostToDevice);    
    cudaMalloc (&fixed_cap, sizeof(int)*k1);
    

    // d_fac_id and fixed_fac_id were created to fix the appended zeros in the facility array.
    int *d_fac_id, *fixed_fac_id;
    cudaMalloc (&d_fac_id, sizeof(int)*N*max_P);
    cudaMemcpy (d_fac_id, fac_ids, sizeof(int)*N*max_P, cudaMemcpyHostToDevice);    
    cudaMalloc (&fixed_fac_id, sizeof(int)*k1);
    
    
    // To update and store the number of successful requests per center.
    int *d_succ_reqs;
    cudaMalloc (&d_succ_reqs, sizeof(int)*N);
    cudaMemset (d_succ_reqs, 0, sizeof(int)*N);
    
    cudaDeviceSynchronize();
    //*********************************
    // Call the kernels here
    

    // kernel calls to fix capacity arrays and facility arrays.
    fixCapacity<<<ceil(k1/1024.0), 1024>>> (fixed_cap, d_cap, k1);
    fixFac<<<ceil(k1/1024.0), 1024>>> (fixed_fac_id, d_fac_id, k1);
    cudaFree(d_cap);
    cudaFree(d_fac_id);

    
    // generate the bucket sizes.
    generateOffset<<<ceil(R/1024.0), 1024>>> (d_count, d_offset, d_req_cen, d_req_fac, R);
    preSum(f_offset, d_count, k1);
    
    
    // push requests into the csr list.
    pushRequests<<<ceil(R/1024.0), 1024>>> (d_offset, f_offset, f_csr, d_helper, d_req_cen, d_req_fac, R);    


    // sort Requests by unique facility room in csr list.
    sortRequests<<<ceil(k1/1024.0), 1024>>> (d_count, f_offset, f_csr, d_req_cen, d_req_fac, k1);
    
    
    cudaFree ((int*)d_helper);
    scheduleRequests<<<ceil(k1/1024.0), 1024>>> (d_offset, fixed_cap, d_count, f_offset, f_csr, d_req_cen, d_req_fac, d_req_start, d_req_slots, d_succ_reqs, k1);
    
    //********************************

    /*printArr<<<1,1>>> (d_count, k1);
    printArr<<<1,1>>> (d_offset, N);
    printArr<<<1,1>>> (fixed_fac_id, k1);
    printArr<<<1,1>>> (fixed_cap, k1);
    printArr<<<1,1>>>(f_offset, k1);
    printArr<<<1,1>>>(f_csr, R);
    printArr<<<1,1>>> (d_succ_reqs, N);*/
    
    cudaMemcpy (succ_reqs, d_succ_reqs, sizeof(int)*N, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    for (int i = 0; i < N; i++) {
        success += succ_reqs[i];
    }
    
    fail = R - success;

    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}