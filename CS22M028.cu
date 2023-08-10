/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
/* 
 * This implementation has 4 kernel functions that do the following:
 * 1. Initialize the first level of the graph (0th). This ensures that the active list is updated and we obtain the first 
 * lebel's indices.
 * 
 * 2. processNodes(int* , int* , int* , int*, int*): This updates the active indegree of each node in a particular level.
 * 
 * 3. reduction(int* , int* , int*) :  Find maximum in the csr list pointed to by the offsets of the indices. This
 * primarily helps with updating the indices to point to that of the next level's
 * 
 * 4. activateNodes(int* , int* , int*, int*, int*, int*, int, int): This activates the nodes with active in degree greater than 
 * the activation requirement.
 * 
 * 5. deactivateNodes(int* , int* , int*, int*, int*, int*, int, int): This deactivates the nodes with inactive neighbors and updates the dlevel
 * array.
 * 
*/


__global__ void initializeRoot(int* d_offset, int* d_csrList, int* d_indices, int* d_active, int* d_level, int* d_apr, int V){
    /*
     * This kernel uses the property of the problem that elements in the first level are all non zero and there is only one element
     * with d_apr[itself] = 0 and d_apr[its immediate right] != 0.
     * 
     * This has to be the case as the problem statement states that:
     * 1. The graph passed through the input is level order sorted
     * 2. edges can go from nodes in one level to only the nodes in the next level.
     * 3. d_apr[0th level elements] = 0 and d_apr[non 0th level elements] != 0.
     * 
    */
    
    int glob_id = blockIdx.x*blockDim.x+threadIdx.x; // obtaining the global ID of this particular thread.
    
    if(glob_id > V) return; // to avoid grabage values due to unnecessary threads launched.
    
    if(d_apr[glob_id] == 0 and (glob_id+1) < V and d_apr[glob_id+1] != 0){
        d_indices[1] = glob_id; // update indices first.
    }
    if(d_apr[glob_id] == 0 and d_active[glob_id] == 0){
        d_active[glob_id] = 1; // update the active lists later on.
    }
    d_level[0] = d_indices[1] + 1; // update the d_level[0] : to get the number of active nodes in the 0th level.
}

__global__ void sanityCheck(int* d_apr, int* d_indices, int V){
    if(d_indices[1] == 0){
        if(d_apr[0] == 0){
            d_indices[1] = V-1;
        }else{
            d_indices[1] = 0;
        }
    }
}

__global__ void processNodes(int *d_offset, int* d_csr, int* d_indices,int* d_aid, int* d_active){
    /*
     * 1. This kernel basically updates the active indegrees of the kernels. 
     * 2. It is launched with appropriate configuration, it accesses the csr list ot get update the indegrees of the nodes in the next
     * level.
     * 3. In order to avoid cache incoherence, further processing on d_aid is avoided. Also atomics do not support volatile variables.
     * 
    */
    
    int start = d_indices[0]; // load in the indices to the registers.
    int end = d_indices[1]; // load in the indices to the registers.
    
    int glob_id = blockIdx.x*blockDim.x+threadIdx.x; // calculation of the global ID of the thread.
    
    if(glob_id+start > end) return; // To avoid invalid data due to extra threads launched.
    
    if(d_active[glob_id+start] != 0) // update only for nodes in this level that are active.
    {
        for(int i = d_offset[glob_id+start]; i < d_offset[glob_id+start+1]; i++)
        {
            int index = d_csr[i]; // for smooth atomic ops, import global memory data into registers.
            int* point = d_aid+index;
            atomicAdd(point,1); // update d_aid.
        }
    }
}

__global__ void activateNodes(int *d_offset, int* d_indices, int* d_aid, int* d_apr, int* d_active, int* d_level, int i, int V){
    /*
     * This kernel activates nodes based only on whether the active indegree >= activation point requirement.
     * This kernel, does not deactivate any nodes and the deactivation is left to another kernel.
     */
    int start = d_indices[0]; //get starting index into a register.
    int end = d_indices[1]; // get ending index into a register.
    
    int glob_id = blockIdx.x*blockDim.x+threadIdx.x; //calculate global ID of the thread.
    
    int node = (glob_id)+start; // calculate index of the corresponding node.
    
    if(node > end) return; // avoid segmentation errors/ out of bound errors.

    if(d_aid[node] >= d_apr[node]){ 
        d_active[node] = 1; //activate the node.
        int* point= d_level+i; //atomics may fail if they have to dereference.
        atomicAdd(point, 1); // update the levels.
    }
}

__global__ void deactivateNodes(int *d_offset, int* d_indices, int* d_aid, int* d_apr, int* d_active, int* d_level, int i, int V){

    /*
     * The deactivation of nodes which have their immediate neighbours existant and deactivated are take care of here.
     * The end of the kernel acts as a global barrier to avoid cache incoherence.
    */

    int start = d_indices[0];
    int end = d_indices[1];
    
    int glob_id = blockIdx.x*blockDim.x+threadIdx.x;
    
    int node = (glob_id) + start;
    
    if(node > end) return;
    
    if(node-1 >= start and d_active[node-1] == 0 and node+1<=end and d_active[node+1] == 0 and d_active[node] == 1) {
        d_active[node] = 0;
        int* point = d_level+i;
        atomicAdd(point, -1);
    }
}

__device__ unsigned answer = 0;

__global__ void reduction(int* d_offset, int* d_csrList, int* d_indices){

    /*
     * This kernel is used to find the end index of the next level.
     * It uses reduction and in order to avoid across block synchronization issuse, it does not update the Indices here.
    */

    extern __shared__ int CSRd[];
    
    int* maxArr = CSRd;
    int temp = d_indices[1];
    
    //((threadIdx.x + blockIdx.x*blockDim.x) + d_offset[d_indices[0]] <= d_offset[d_indices[1]]-1)?maxArr[threadIdx.x] = d_csrList[(blockIdx.x*blockDim.x+threadIdx.x)+d_offset[d_indices[0]]]:maxArr[threadIdx.x] = INT_MIN;
    int glob_id = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(glob_id + d_offset[d_indices[0]] <= (d_offset[d_indices[1]]-1)){
        maxArr[threadIdx.x] = d_csrList[glob_id+d_offset[d_indices[0]]];
    }else{
        maxArr[threadIdx.x] = INT_MIN;
    }
    __syncthreads();
    
    int off = blockDim.x;
    
    for(int i = ceil(float(blockDim.x)/2); i >= 1; i/=2){
        if(threadIdx.x+i < off){
            int s1 = threadIdx.x; int s2;
            (threadIdx.x+i<blockDim.x)?s2 = threadIdx.x+(i): s2 = s1;
            int element1 = maxArr[s1];
            int element2 = maxArr[s2];
            maxArr[s1] = element1>element2?element1:element2;
            s1 == s2? maxArr[s1] = element1:element2;
        }
        off = ceil(off/2.0);
        __syncthreads();
    }
    if(threadIdx.x == 0){
        if(blockDim.x%2){
            maxArr[0] = (maxArr[0]<maxArr[blockDim.x-1])?maxArr[0]:maxArr[blockDim.x-1];
        }
    }
    
    if(threadIdx.x == 0){
        int curr = maxArr[0];
        atomicMax(&answer, curr);
    }
}
__global__ void updateIndices(int* d_indices){
    /*
     * This kernel actually updates the indices to point to the next level of the graph.
    */
    d_indices[0] = d_indices[1]+1;
    d_indices[1] = answer;
    answer = 0;
}

__global__ void printKernel(int* d_aid, int* d_indices){
    for(int i = d_indices[0]; i <= d_indices[1]; i++){
        printf("%d ", d_aid[i]);
    }printf("\n");
}
__global__ void printIndices(int* d_indices){
    printf("On GPU : %d %d\n", d_indices[0], d_indices[1]);    
}
/**************************************END*************************************************/

//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc, char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments
    
cudaMemset(d_aid, 0 , V*sizeof(int));

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

// defining auzilliary data structures for processing data
int* d_active, * d_indices, *d_level;
cudaMalloc(&d_active, V*sizeof(int)); // allocate memory for device version of active array.
cudaMalloc(&d_level, L*sizeof(int)); // allocate memory for device version of level array.
cudaMalloc(&d_indices, 2*sizeof(int)); // allocate memory for device version of indices array.
cudaMemset(d_level, 0 ,L*sizeof(int)); // initialize device version of level array.
cudaMemset(d_active, 0, V*sizeof(int)); // initialize device version of active array.
cudaMemset(d_indices, 0, 2*sizeof(int)); // initialize device version of indices array.

int indices[] = {0,0};
    
int reqThreads = 10003; // To account for some errors in test Cases.
int threadsPerBlock = 512; // maximum number of threads I would allow/ recently, it has been recommended not to use all in a block due to dynamic allocation.
int numOfBlocks = ceil(float(reqThreads)/threadsPerBlock); // number of blocks in order to cover for all of the required threads.

initializeRoot<<<numOfBlocks, threadsPerBlock>>>(d_offset, d_csrList, d_indices, d_active, d_level, d_apr, V); // initialize the first level of the graph.
//sanityCheck<<<1,1>>>(d_apr, d_indices, V); // in case all the vertices are in the first level of the graph.

cudaMemcpy(indices, d_indices, 2*sizeof(int), cudaMemcpyDeviceToHost); // copy back the calculated indices.

/*
 * For the remaining levels, four kernels are called 
 * processNodes<<<..>>>(..) : This kernel will update the active in degrees of the nodes in the current level(using data from the previous level).
 * reduction<<<..>>>(..) : This kernel will calculate the indices to point to the next level of the graph in the CSR List.
 * updateIndices<<<..>>>(..) : This kernel will update the indices in the device.
 * activateNodes<<<..>>>(..) : This kernel will activate the nodes in the current level if the based on the active indegree and apr only.
 * deactivateNodes<<<..>>>(..) : This kernel will deactivate the nodes in the current level based on the activation status of its immediate existant neighbors.
*/
for(int i = 1; i < L; i++){

    // set threadsPerBlock, reqThreads, numOfBlocks, which represent whatever was already defined but for the kernel about to be called.
    
    // setup of configuration for the launch of processNodes<<<..>>>(..) kernel================================================================>
    threadsPerBlock = 512;
    reqThreads = indices[1]-indices[0]+1; // number of threads to be the same as the number of elements in the current level.
    
    numOfBlocks = ceil((float)reqThreads/threadsPerBlock); // number of blocks required to accommodate the current level.
    
    processNodes<<<numOfBlocks,threadsPerBlock>>>(d_offset, d_csrList, d_indices, d_aid, d_active); // process the active indices of the current level.
    

    // setup of configuration for the launch of reduction<<<..>>>(..) kernel====================================================================>
    threadsPerBlock = 512; 
    reqThreads = h_offset[indices[1]] - h_offset[indices[0]]; // number of threads to be same as the number of edges from current level to next level.
    
    numOfBlocks = ceil((float)reqThreads/threadsPerBlock); // number of blocks required to accomodate the number of edges.
    reduction<<<numOfBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(d_offset, d_csrList, d_indices); // reduce the csr list to get the next ending index.
    indices[0] = indices[1]+1; // update the start index in the CPU only.

    // the config for updateIndices<<<>>>() is simple.
    updateIndices<<<1,1>>>(d_indices); // update the indices in the device.
    
    cudaMemcpy(indices+1, d_indices+1, sizeof(int), cudaMemcpyDeviceToHost); // copy back ending index from device.

    // setup of configuration for the launch of activateNodes<<<..>>>(..) kernel and deactivateNodes<<<..>>>>(..) kernel==========================>
    reqThreads = indices[1]-indices[0]+1;
    threadsPerBlock = 512;
    
    numOfBlocks = ceil((float)reqThreads/threadsPerBlock);
    activateNodes<<<numOfBlocks, threadsPerBlock>>>(d_offset, d_indices, d_aid, d_apr, d_active, d_level, i, V); // call activateNodes<<<..>>>(..).
    deactivateNodes<<<numOfBlocks, threadsPerBlock>>>(d_offset, d_indices, d_aid, d_apr, d_active, d_level, i, V); // call deactivateNodes<<<..>>>(..).
}
cudaDeviceSynchronize();



/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
cudaMemcpy(h_activeVertex, d_level, (L)*sizeof(int), cudaMemcpyDeviceToHost);
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
    cudaError_t err = cudaGetLastError();
    printf("error: %d %s %s \n",err, cudaGetErrorName(err), cudaGetErrorString(err));

for(int i=0; i<L; i++)
{
    printf("level = %d , active nodes = %d\n", i, h_activeVertex[i]);
}
    return 0;
}