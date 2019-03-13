#include<stdio.h>
#include<mpi.h>
#include<unistd.h>
#include<math.h>


int main(int argc, char **argv){
  int rank=0;
  int world_size=0;
  int i,j,k;
  int p;
  int dimension=2;
  int dims[2];
  int periods[2];
  int coordinate[2];
  int temp_coordinate[2];
  double matrix[16][16];
  int rows=0;
  int cols=0;
  double data;
  int tag = 4;
  int where_to = 0;
  int next_rank;
  double computation_time;
  MPI_Comm topo;
  /*Return status for recieve  */
  MPI_Status status;

/* Startup MPI*/	
  MPI_Init(&argc,&argv);
  computation_time = MPI_Wtime();
 /*Find out number of processes */ 
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
/*Find out process rank */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  p = (int)sqrt(world_size);

  for(i=0;i<dimension;i++){
	dims[i] = p;
	periods[i] = 0;
  }
/*Create a 2D mesh Topology */
  MPI_Cart_create(MPI_COMM_WORLD,dimension,dims,periods,0,&topo);
  MPI_Cart_coords(topo,rank,dimension,coordinate);

/*initialize a matrix with random values from 0 to 10 at process0*/
  if(rank ==0){
        for(i = 0;i<p;i++)
                for(j=0;j<p;j++){
                        matrix[i][j] = rand()%10;
                }
  	}
  	if(rank ==0){
  	for(i = 0;i<p;i++){
     		for(j = 0;j<p;j++){
                	printf("%f\t",matrix[i][j]);
        	}
        	printf("\n");
 	}
  }
/*Parallel Formulation for LU decomposition*/
  if(rank ==0)
	data = matrix[0][0];
/* Processor 0 send a matrix on a 2d mesh of processors*/
  for(i= 0; i<p; i++){
  	for(j= 0; j<p; j++){
        	if((i!=0)||(j!=0)){
  			where_to++;
       		        if(rank==0)
 			   	MPI_Send(&matrix[i][j],1,MPI_DOUBLE,where_to,tag,MPI_COMM_WORLD);
           		else if(rank==where_to)
               	 	  	 MPI_Recv(&data,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
            	}
        }
    } 
/*Serial Formulation for LU decomosition  */
/*  for(i=0;i<p;i++){
	for(j=0;j<p;j++){
		if(i ==0 && j ==0){
			if(rank ==0)
				data = matrix[i][j];
		}
		else{
			where_to++;
			if(rank == 0){
				MPI_Send(&matrix[i][j],1,MPI_INT,where_to,tag,MPI_COMM_WORLD);
			}
			if(rank == where_to++){
				MPI_Recv(&data,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
				printf("In rank %d with data %d\n",rank,data);
			}
		}
	}
  }
*/
  for(k= 0; k<p; k++){
	double temp1,temp2,temp3;
	for(j= k+1; j<p; j++){
		temp_coordinate[0]= k;
	        temp_coordinate[1]= j;
           	MPI_Cart_rank(topo,temp_coordinate,&next_rank);
        	if((coordinate[0]==k)&&(coordinate[1]==k)){
			MPI_Send(&data,1,MPI_DOUBLE,next_rank,tag,topo);
            	}
                else if((coordinate[0]==k)&&(coordinate[1]==j)){
			temp_coordinate[0]= k;
     		        temp_coordinate[1]= k;
                	MPI_Cart_rank(topo,temp_coordinate,&next_rank);
       	                MPI_Recv(&temp1,1,MPI_DOUBLE,next_rank,tag,topo,&status);
            		data = data/temp1;
                 }
        }
	for(i= k+1; i<p; i++){
		for(j= k+1; j<p; j++){
                	if((coordinate[0]==i)&&(coordinate[1]==k)){
                  		  temp_coordinate[0]= i;
          		          temp_coordinate[1]= j;
            	      	          MPI_Cart_rank(topo,temp_coordinate,&next_rank);
                  		  MPI_Send(&data,1,MPI_DOUBLE,next_rank,tag,topo);
              		  }
               		 else if((coordinate[0]==i)&&(coordinate[1]==j)){
				temp_coordinate[0]= i;
               		        temp_coordinate[1]= k;
   		                MPI_Cart_rank(topo,temp_coordinate,&next_rank);
               		        MPI_Recv(&temp2,1,MPI_DOUBLE,next_rank,tag,topo,&status);
              		  }
           	 }
        }
        for(i= k+1; i<p; i++){
		for(j= k+1; j<p; j++){
			if((coordinate[0]==k)&&(coordinate[1]==j)){
	                        temp_coordinate[0]= i;
            		        temp_coordinate[1]= j;
                   	        MPI_Cart_rank(topo,temp_coordinate,&next_rank);
             		        MPI_Send(&data,1,MPI_DOUBLE,next_rank,tag,topo);
               		 }
               		 else if((coordinate[0]==i)&&(coordinate[1]==j)){
          		        temp_coordinate[0]= k;
               	         	temp_coordinate[1]= j;
               		        MPI_Cart_rank(topo,temp_coordinate,&next_rank);
               		        MPI_Recv(&temp3,1,MPI_DOUBLE,next_rank,tag,topo,&status);
               		        data = data - (temp2*temp3);
            		 }
            	}
        }
  }

  for(k= p-1; k>0; k--){
	double temp;
	if((coordinate[0]==k)&&(coordinate[1]==k)){
 		temp_coordinate[0]= k-1;
            	temp_coordinate[1]= k-1;
         	MPI_Cart_rank(topo,temp_coordinate,&next_rank);
            	MPI_Send(&data,1,MPI_DOUBLE,next_rank,tag,topo);
        }
        else if((coordinate[0]==k-1)&&(coordinate[1]==k-1)){
                temp_coordinate[0]= k;
            	temp_coordinate[1]= k;
            	MPI_Cart_rank(topo,temp_coordinate,&next_rank);
            	MPI_Recv(&temp,1,MPI_DOUBLE,next_rank,tag,topo,&status);
            	data = data*temp;
        }
  }
  if((coordinate[0]==0)&&(coordinate[1]==0)){
	printf("Determinant of the matrix after LU decomposition of the initial matrix is: %f \n",data);
     	//Measure the Computation time
	computation_time = MPI_Wtime() - computation_time;
     	printf("Total computation time required to find the determinant of the matrix is: %f seconds \n",computation_time);
  }
  MPI_Finalize(); //Finalizing MPI
  return 0;
}
