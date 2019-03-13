#include<stdio.h>
#include<unistd.h>
#include<mpi.h>
#include<math.h>

#define ARR_SIZE 16 /*Globally declare the array size*/

/*Compute the XOR between the two processot id*/
int xor(int a, int b){
	return a ^ b;
}

/*Compute logical and operation between the two ids of processor*/
int and(int a,int b){
	return a & b;
}

/*Main begins*/
int main(int argc, char **argv){
double t1_comp=0;
double t2_comp=0;
double communication_time = 0;
double t1_comm=0;
int c1,p1,b,c,d1,e,i,k,j,p,N1,mask,mask1,dest,dest1,source1,source;  
int world_size;
int rank;
MPI_Status status; /*Receive status flag*/
int tag =4;
int master = 0;
int q = 3;
int d = 3; /*Dimesnion of the hypercube*/
int chunk_size;
int offset = 0;
int sum = 0;
int value=0;
int partial_sum = 0;
int a[ARR_SIZE];

/*Initializion*/
  MPI_Init(&argc,&argv);

/*Find the number of processors*/
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

/*Find the rank of each processor*/
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
/*Normalising the array to chunks of fixed size*/
  chunk_size = ARR_SIZE/world_size;
  offset = chunk_size*rank;

/*Populate the array randomly with numbers */
  if(rank == master){
  	for (i=0;i<ARR_SIZE;i++)
  	{
  	a[i]=rand()%10;
  }}

/*One to all broadcast to find the communication time where all processors get the messages finally*/
 mask = (int)pow((double)2,d)-1;
 for(i = d-1;i>=0;i--){
/*Set bit i of mask to 0*/
	mask = xor(mask,(int)pow((double)2,i));
/*Check if th current processor is active or not*/	
if(and(rank, mask) == 0)
{
/*Check if the processor is a sender */
		if(and(rank,(int) pow((double)2,i)) == 0){
			dest = xor(rank,(int) pow((double)2,i));
			t1_comm = MPI_Wtime();
			MPI_Send(a,ARR_SIZE,MPI_INT,dest,tag,MPI_COMM_WORLD);
		}
		else
/*The processor is a receiver*/
{
			source = xor(rank,(int)pow((double)2,i));
			MPI_Recv(a,ARR_SIZE,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
			/*Calculate communication time*/
			communication_time = communication_time+(MPI_Wtime() - t1_comm);
		}

	}
}

/*Print communication time when the rank of the proecessor is the last one i.e (n-1)*/
if(rank==world_size-1)
{
printf("The communication time is %lf picoseconds\n",communication_time);	
}

/*Single Node Accumulation for finding the sum of all numbers in the array*/
  mask1 = 0;
/*Time starts for computing the sum*/
t1_comp=MPI_Wtime();
    for (j = 0; j<chunk_size; j++)
    {
 	sum = sum + a[j + chunk_size*rank];
    }
    for (p= 0; p<=d-1; p++)
    { 
/*Select processors whose lower p bits are 0*/ 
 c1 = pow(2,p);
        if((mask1&rank)==0)
        {
            if((rank&c1)!=0)
            {
                c = pow(2,p);
                dest1 = rank^c;
                MPI_Send(&sum,1,MPI_INT,dest1,0,MPI_COMM_WORLD);
            }
            else
            {
                d1= pow(2,p);
                source1 = rank^d1;
                MPI_Recv(&value,1,MPI_INT,source1,0,MPI_COMM_WORLD,&status);
                sum = sum + value;
		/*Time ends to compute the sum*/
		t2_comp=MPI_Wtime();
            }
         }
	    e= pow(2,p);
/*Set bit p of mask to 1*/
            mask1 = mask1^e;
    }

   if(rank==0)
  {
   printf("The total number of processors is %d \n", world_size);
   printf("The sum of the array is %d\n",sum);
   printf("Total Computaion time is %lf\n",t2_comp-t1_comp); 
 }
MPI_Finalize();
return 0;
}
