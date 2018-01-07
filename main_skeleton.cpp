//! @file Simple 2D Heat Diffusion simulator

#include <mpi.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

#include <string.h>
#include <math.h>
#include <tuple>
#include <iostream>

using namespace std;

// minimum iteration difference precision
#define EPSILON 0.00001

/// @struct Spot
/// @brief The struct with parameters of a spot with given temperature
struct Spot {
    unsigned int x; ///< X-coordination of the spot
    unsigned int y; ///< y-coordination of the spot
    unsigned int temperature; ///< temperature of the spot

    /// operator== Comparison of spots from coordination point of view
    /// @param b - Spot for comparation
    bool operator==(const Spot& b) const
    {
        return (this->x == b.x) && (this->y == b.y);
    }
};

/**
 * @param n number of given values
 * @param sum sum of values
 * @return mean
 */
float getMean(int n, float sum);

/// readInstance - Method for reading the input instance file
/// @param instanceFileName - File name of the input instance
/// @return Tuple of (Width of the space; Height of the Space; Vector of the Spot)
tuple<unsigned int, unsigned int, vector<Spot>> readInstance(const char *instanceFileName) {
    unsigned int width, height;
    vector<Spot> spots;
    string line;

    ifstream file(instanceFileName);
    if (file.is_open()) {
        int lineId = 0;
        while (std::getline(file, line)) {
            stringstream ss(line);
            if (lineId == 0) {
                ss >> width;
            }
            else if (lineId == 1) {
                ss >> height;
            }
            else {
                unsigned int i, j, temperature;
                ss >> i >> j >> temperature;
                spots.push_back({i, j, temperature});
            }
            lineId++;
        }
        file.close();
    }
    else {
        throw runtime_error("It is not possible to open instance file!\n");
    }
    return make_tuple(width, height, spots);
}

/// writeOutput - Method for creating resulting ppm image
/// @param myRank - Rank of the process
/// @param width - Width of the 2D space (image)
/// @param height - Height of the 2D space (image)
/// @param image - Linearized image
void writeOutput(const int myRank, const int width, const int height, const string instanceFileName, const float *image){//vector<int> image){
    // Draw the output image
    ofstream file(instanceFileName);
    if (file.is_open())
    {
        if (myRank == 0) {
            file << "P2\n" << width << "\n" << height << "\n" << 255 <<  "\n";
            for (unsigned long i = 0; i < width*height; i++) {
                file << static_cast<int> (image[i]) << " ";
            }
        }
    }
    file.close();
}

int filterOnInner(float * block, float * newBlock, int block_width, int block_height){

    int different = 0; //1 if blocks are still not convergent, 0 if we done
    for (int y = 2; y < block_height - 2; ++y) {
        for (int x = 0; x < block_width; ++x) {
            if(newBlock[y*block_width+x]>=0){//if(isSpot(x,(y-1)+(block_height-2)*rank,spots,spotSize,block_width,block_height)) {//(block_height-2)
                //newBlock[y*block_width+x] = block[y*block_width+x];
                continue;
            }
            int divide = 9;
            float sum = 0;
            //do 3 common to all
            sum+=block[y*block_width + x];
            sum+=block[(y-1)*block_width + x];
            sum+=block[(y+1)*block_width + x];
            if (x==0) {//if start of line
                divide = 6;

            }else{
                //do 3 on the left
                sum+=block[y*block_width + x-1];
                sum+=block[(y-1)*block_width + x-1];
                sum+=block[(y+1)*block_width + x-1];
            }
            if (x==block_width-1) divide = 6;
            else{
                sum+=block[y*block_width + x+1];
                sum+=block[(y-1)*block_width + x+1];
                sum+=block[(y+1)*block_width + x+1];
            }

            float mean = getMean(divide, sum);
            newBlock[y*block_width+x] = mean;
            if(fabs(mean-block[y*block_width + x])>=EPSILON) different = 1;//there is still a point to convolute
        }

    }
    return different;
}

/**
 * Processes data sent from neighbours block nodes.
 * @param block block
 * @param new_block new block
 * @param block_width block width
 * @param block_height block height
 * @param rank node rank
 * @param number_of_nodes number of nodes
 * @return change between iterations
 */
int filterOnReceived(float * block, float * new_block, int block_width, int block_height, int rank, int number_of_nodes){
    int  number_of_counted = 0;
    int y = 1;
    int change = 0;
    float sum = 0;

    for (int x = 0; x < block_width; ++x) {
        if(new_block[y*block_width+x] >= 0){
            continue;
        }
        number_of_counted=0;
        sum=0;

        for (int y1 = -1; y1 < 2; ++y1) {
            if(rank == 0 && ( y1==-1)) {
                continue;
            }
            for (int x1 = -1; x1 < 2; ++x1) {
                if(( x == 0 && x1 == -1 ) || ( x == block_width -1 && x1 == 1)) {
                    continue;
                }
                number_of_counted++;
                sum+=block[(y+y1)*block_width + x+x1];

            }
        }

        float mean = getMean(number_of_counted, sum);
        new_block[y*block_width+x] = mean;

        // still need to iterate
        if(fabs(mean - block[y*block_width + x]) >= EPSILON) {
            change = 1;
        }
    }

    //if(rank == number_of_nodes-1 && ( y1==1)) continue;

    y=block_height-2;
    for (int x = 0; x < block_width; ++x) {
        if(new_block[y*block_width+x]>=0){
            continue;
        }
        number_of_counted=0;
        sum=0;

        for (int y1 = -1; y1 < 2; ++y1) {
            if(rank == number_of_nodes-1 && ( y1==1)) continue;
            for (int x1 = -1; x1 < 2; ++x1) {
                if((x==0 && x1==-1) || (x == block_width - 1 && x1 == 1)) {
                    continue;
                }
                number_of_counted++;
                sum+=block[(y+y1)*block_width + x+x1];

            }
        }

        float mean = getMean(number_of_counted, sum);
        new_block[y*block_width+x] = mean;

        // still need to iterate
        if(fabs(sum-block[y*block_width + x])>=EPSILON) {
            change = 1;
        }
    }

    return change;

}

float getMean(int n, float sum) {
    float mean = sum / (float) n;
    if ( mean < 255 ) {
        return mean;
    } else {
        return 255;
    }
}

/*
 * Method that distributes the spots given by x, y and temperature to the block on the right x, y coordinate.
 */
void distributeSpots(int rank, float *block, Spot *spots, int spotSize, int block_width, int block_height){
    for (int i = 0; i < spotSize; ++i) {
        int block_start = block_height*rank;
        if  (spots[i].y >= block_start && spots[i].y < block_start + block_height){
            block[((spots[i].y + 1 - block_start) * block_width + spots[i].x)] = spots[i].temperature;
        }
    }
}

/**
 * Swaps two pointers
 * @param one pointer
 * @param another pointer
 */
void swap_pointers(float *one, float *another) {
    float *foo = one;
    one = another;
    another = foo;
}

/**
 * Iterates algorithm for heat difussion.
 * @param rank cpu rank
 * @param number_of_nodes number of nodes
 * @param block assigned block
 * @param block_width widht of assigned block
 * @param block_height height of assigned block
 * @param spots all spots
 * @param number_of_spots number_of_slots
 * @return
 */
float *iterate(int rank, int number_of_nodes, float *block, int block_width, int block_height, Spot *spots, int number_of_spots){
    MPI_Request reqSendUpper, reqRecvUpper,reqSendLower, reqRecvLower;
    int block_size = block_width * (block_height+2);
    auto new_block = (float *) malloc(block_size * sizeof(float));

    while(1){
        for (int i = 0; i < block_width*(block_height+2); ++i) {
            new_block[i]=-1;
        }
        if (rank!=0){
            MPI_Isend(block+block_width, block_width,MPI_FLOAT, rank-1,1,MPI_COMM_WORLD, &reqSendUpper);
            MPI_Irecv(block, block_width,MPI_FLOAT, rank-1,1,MPI_COMM_WORLD, &reqRecvUpper);

        }
        if (rank < number_of_nodes-1) {
            MPI_Isend(block + block_width * block_height, block_width,MPI_FLOAT, rank+1,1,MPI_COMM_WORLD, &reqSendLower);
            MPI_Irecv(block + block_width * block_height + block_width, block_width,MPI_FLOAT, rank+1,1,MPI_COMM_WORLD, &reqRecvLower);
        }

        int endOfConvolution = 0; //0 done, 1 not yet done

        distributeSpots(rank, new_block, spots, number_of_spots, block_width, block_height);
        endOfConvolution |= filterOnInner(block,new_block, block_width,block_height+2);

        if (rank < number_of_nodes-1) {
            MPI_Wait(&reqRecvLower, nullptr);
        }
        if (rank!=0) {
            MPI_Wait(&reqRecvUpper, nullptr);
        }

        endOfConvolution |= filterOnReceived(block,new_block,block_width,block_height+2,rank,number_of_nodes);

        int globalEnd = 0;
        MPI_Allreduce(&endOfConvolution, &globalEnd, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(globalEnd==0) {
            // no change - end
            break;
        }

        if(rank < number_of_nodes -1) MPI_Wait(&reqSendLower, nullptr);
        if(rank!=0) MPI_Wait(&reqSendUpper, nullptr);

        swap_pointers(new_block, block);

        for (int i = 0; i < block_width*(block_height+2); ++i) {
            cout << block[i] << " ";
            if ( i % block_width == 0 ) {
                cout << "\n";
            }
        }
        cout << "\n";

        for (int i = 0; i < block_width*(block_height+2); ++i) {
            cout << new_block[i] << " ";
            if ( i % block_width == 0 ) {
                cout << "\n";
            }
        }
        cout << "\n";
    }

    // free
    free(new_block);
    return block;
}


/**
 * Cast spots to MPI_Datatype.
 * @return
 */
MPI_Datatype createSpotType(){
    int lenghts_of_block[3]={1,1,1};
    MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};
    MPI_Datatype mpi_spot;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Spot, x);
    offsets[1] = offsetof(Spot, y);
    offsets[2] = offsetof(Spot, temperature);
    MPI_Type_create_struct(3,lenghts_of_block,offsets,types,&mpi_spot);
    MPI_Type_commit(&mpi_spot);
    return mpi_spot;
}


/**
 * Main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Initialize MPI
    int worldSize, myRank;
    int initialised;
    MPI_Initialized(&initialised);
    if(!initialised)
        MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Datatype mpi_spot = createSpotType();

    if (argc > 1) {
        // read the input instance
        unsigned int width, height, spotsSize, blockHeight;
        vector<Spot> spots;
        Spot *spotData;
        float * output; // linearized output

        //-----------------------\\
        // Insert your code here \\
        //        |  |  |        \\
        //        V  V  V        \\
        // Setup openMPI task
        if (myRank == 0) {
            tie(width, height, spots) = readInstance(argv[1]);
            spotsSize = static_cast<unsigned int>(spots.size());

            // round the spacesize of the output to be multiple of the number of used processes
            if (height % worldSize != 0) {
                height = ((height / worldSize) + 1) * worldSize;
            }
            if (width % worldSize != 0) {
                width = ((width / worldSize) + 1) * worldSize;
            }

            // data chunks in rows
            blockHeight = height / worldSize;

            output = new float[ width * height];
            memset(output,'\0',width*height*4);//memset works with bytes FIXME

            for (Spot &spot : spots) {
                int index = spot.y*width + spot.x;
                output[index] = spot.temperature;
            }

            auto *parameters = new unsigned int[3];
            parameters[0] = width;
            parameters[1] = blockHeight;
            parameters[2] = spotsSize;

            int enlargedBlockHeight = (blockHeight+2)*width;//make some space for rows from other processors FIXME

            auto buffer = (float *)malloc(enlargedBlockHeight*sizeof(float));//new int [enlargedBlockHeight];
            memset(buffer,'\0',enlargedBlockHeight*4);
            MPI_Bcast(parameters,3,MPI_INT,0,MPI_COMM_WORLD);
            spotData = spots.data();
            MPI_Bcast(spotData,spotsSize,mpi_spot,0,MPI_COMM_WORLD);
            distributeSpots(myRank, buffer, spotData, spotsSize, width, blockHeight);
            //instead of Scatter, Bcast the Spots
            buffer = iterate(myRank, worldSize, buffer, width, blockHeight, spotData, spotsSize);
            MPI_Gather(buffer+width,width*blockHeight,MPI_FLOAT,output,width*blockHeight,MPI_FLOAT,0,MPI_COMM_WORLD);
            free(buffer);
            delete(parameters);
        }else{
            int * tmpWH = new int[3];;

            MPI_Bcast(tmpWH,3,MPI_INT,0,MPI_COMM_WORLD);
            int block_width = tmpWH[0], block_height = tmpWH[1];
            spotsSize = tmpWH[2];
            int size = block_width * (block_height+2);
            auto blockBuff = (float *)malloc(size*sizeof(float));
            spotData = (Spot *)malloc(spotsSize* sizeof(Spot));
            MPI_Bcast(spotData,spotsSize,mpi_spot,0,MPI_COMM_WORLD);

            distributeSpots(myRank, blockBuff, spotData, spotsSize, block_width, block_height);
            blockBuff = iterate(myRank, worldSize, blockBuff, block_width, block_height, spotData, spotsSize);
            MPI_Gather(blockBuff+block_width,block_width*block_height,MPI_FLOAT,output,block_width*block_height,MPI_FLOAT,0,MPI_COMM_WORLD);

            free(blockBuff);
            delete(tmpWH);
        }

        //-----------------------\\

        if(myRank == 0) {
            string outputFileName(argv[2]);
            writeOutput(myRank, width, height, outputFileName, output);
        }
    }
    else {
        if (myRank == 0)
            cout << "Input instance is missing!!!\n" << endl;
    }
    MPI_Finalize();
    return 0;
}