#include<math.h>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>

#include<dirent.h>
#include<errno.h>
#include<string.h>


#define FREE_ARG char*
#define NR_END 1

#define niterations 100
#define fillincrement 0.01
#define oneoversqrt2 0.707106781186

double **topo,**itopo,**toposave,*topovec,**depth,**slope,**slope3,**maxslope,**avgslope,**volume,**flow,**flow1,**flow2,**flow3;
double **flow4,**flow5,**flow6,**flow7,**flow8,dx;
int nx,ny,*iup,*idown,*jup,*jdown,*iup3,*idown3,*jup3,*jdown3;

double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(double)));
        return v-nl+NR_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
        int *v;

        v=(int *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(int)));
        return v-nl+NR_END;
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_vector(double *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

int **imatrix(int nrl, int nrh, int ncl, int nch)
/* allocate an int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
      int  i,**m;

       /*allocate pointers to rows */
        m=(int **)malloc((unsigned) (nrh-nrl+1)*sizeof(int*));
      m -= nrl;

       /*allocate rows and set pointers to them */
        for(i=nrl;i<=nrh;i++) {
                      m[i]=(int *)malloc((unsigned) (nch-ncl+1)*sizeof(int));
      m[i]-=ncl;}
       /* return pointer to array of pointers to rows */
        return m;
}

double **matrix(int nrl, int nrh, int ncl, int nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    int i;
    double **m;

        /*allocate pointers to rows */
        m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
    m -= nrl;

   /*allocate rows and set pointers to them */
      for(i=nrl;i<=nrh;i++) {
                      m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double)
);
            m[i] -= ncl;
    }
      /* return pointer to array of pointers to rows */
      return m;
}

#define SWAP(a,b) itemp=(a);(a)=(b);(b)=itemp;
#define M 7
#define NSTACK 100000

void indexx(int n, double arr[], int indx[])
{
        unsigned long i,indxt,ir=n,itemp,j,k,l=1;
        int jstack=0,*istack;
        double a;

        istack=ivector(1,NSTACK);
        for (j=1;j<=n;j++) indx[j]=j;
        for (;;) {
                if (ir-l < M) {
                        for (j=l+1;j<=ir;j++) {
                                indxt=indx[j];
                                a=arr[indxt];
                                for (i=j-1;i>=1;i--) {
                                        if (arr[indx[i]] <= a) break;
                                        indx[i+1]=indx[i];
                                }
                                indx[i+1]=indxt;
                        }
                        if (jstack == 0) break;
                        ir=istack[jstack--];
                        l=istack[jstack--];
                } else {
                        k=(l+ir) >> 1;
                        SWAP(indx[k],indx[l+1]);
                        if (arr[indx[l+1]] > arr[indx[ir]]) {
                                SWAP(indx[l+1],indx[ir])
                        }
                        if (arr[indx[l]] > arr[indx[ir]]) {
                                SWAP(indx[l],indx[ir])
                        }
                        if (arr[indx[l+1]] > arr[indx[l]]) {
                                SWAP(indx[l+1],indx[l])
                        }
                        i=l+1;
                        j=ir;
                        indxt=indx[l];
                        a=arr[indxt];
                        for (;;) {
                                do i++; while (arr[indx[i]] < a);
                                do j--; while (arr[indx[j]] > a);
                                if (j < i) break;
                                SWAP(indx[i],indx[j])
                        }
                        indx[l]=indx[j];
                        indx[j]=indxt;
                        jstack += 2;
                        if (ir-i+1 >= j-l) {
                                istack[jstack]=ir;
                                istack[jstack-1]=i;
                                ir=j-1;
                        } else {
                                istack[jstack]=j-1;
                                istack[jstack-1]=l;
                                l=i;
                        }
                }
        }
        free_ivector(istack,1,NSTACK);
}
#undef M
#undef NSTACK
#undef SWAP

void insertsortx(int n, double arr[], int indx[])
// Insertion sort using an index table, built to accept an initial guess in indx[]; sorts in place on indx[]
{
	int i,j,indxt;
	double a;
	//for (j=1;j<=n;j++) indx[j]=j;
	for (j=2;j<=n;j++)
	{//Pick out each element in turn.
		indxt=indx[j];
		a=arr[indxt];
		i=j-1;
		while (i > 0 && arr[indx[i]] > a) 
		{ //Look for the place to insert it.
			indx[i+1]=indx[i];
			i--;
		}
		indx[i+1]=indxt; //Insert it.
	}
}

void setupgridneighbors()
{    int i,j;

     idown=ivector(1,nx);
     iup=ivector(1,nx);
     jup=ivector(1,ny);
     jdown=ivector(1,ny);
     for (i=1;i<=nx;i++)
      {idown[i]=i-1;
       iup[i]=i+1;}
     idown[1]=1;
     iup[nx]=nx;
     for (j=1;j<=ny;j++)
      {jdown[j]=j-1;
       jup[j]=j+1;}
     jdown[1]=1;
     jup[ny]=ny;
}

void setupgridneighbors3()
{    int i,j;

     idown3=ivector(1,nx);
     iup3=ivector(1,nx);
     jup3=ivector(1,ny);
     jdown3=ivector(1,ny);
     for (i=1;i<=nx;i++)
      {idown3[i]=i-3;
       iup3[i]=i+3;}
     idown3[3]=1;
	 idown3[2]=1;
	 idown3[1]=1;
     iup3[nx]=nx;
	 iup3[nx-1]=nx;
	 iup3[nx-2]=nx;
	 iup3[nx-3]=nx;
     for (j=1;j<=ny;j++)
      {jdown3[j]=j-3;
       jup3[j]=j+3;}
     jdown3[3]=1;
	 jdown3[2]=1;
	 jdown3[1]=1;
     jup3[ny]=ny;
	 jup3[ny-1]=ny;
	 jup3[ny-2]=ny;
	 jup3[ny-3]=ny;
}

void fillinpitsandflats(int i, int j, double nanval)
{   double min;

   min=topo[j][i];
    if (topo[j][iup[i]]<min) min=topo[j][iup[i]];
    if (topo[j][idown[i]]<min) min=topo[j][idown[i]];
    if (topo[jup[j]][i]<min) min=topo[jup[j]][i];
    if (topo[jdown[j]][i]<min) min=topo[jdown[j]][i];
    if (topo[jup[j]][iup[i]]<min) min=topo[jup[j]][iup[i]];
    if (topo[jup[j]][idown[i]]<min) min=topo[jup[j]][idown[i]];
    if (topo[jdown[j]][idown[i]]<min) min=topo[jdown[j]][idown[i]];
    if (topo[jdown[j]][iup[i]]<min) min=topo[jdown[j]][iup[i]];
    if ((topo[j][i]<=min)&&(i>1)&&(j>1)&&(i<nx)&&(j<ny))
     {topo[j][i]=min+fillincrement;
      if (topo[j][i]!=nanval) fillinpitsandflats(i,j,nanval);
      if (topo[j][iup[i]]!=nanval) fillinpitsandflats(iup[i],j,nanval);
      if (topo[j][idown[i]]!=nanval) fillinpitsandflats(idown[i],j,nanval);
      if (topo[jup[j]][i]!=nanval) fillinpitsandflats(i,jup[j],nanval);
      if (topo[jdown[j]][i]!=nanval) fillinpitsandflats(i,jdown[j],nanval);
      if (topo[jup[j]][iup[i]]!=nanval) fillinpitsandflats(iup[i],jup[j],nanval);
      if (topo[jup[j]][idown[i]]!=nanval) fillinpitsandflats(idown[i],jup[j],nanval);
      if (topo[jdown[j]][idown[i]]!=nanval) fillinpitsandflats(idown[i],jdown[j],nanval);
      if (topo[jdown[j]][iup[i]]!=nanval) fillinpitsandflats(iup[i],jdown[j],nanval);}
}

void mfdflowroute(int i, int j)
{    double tot;
 
     tot=0;
     if (topo[j][i]>topo[j][iup[i]]) 
      tot+=pow(topo[j][i]-topo[j][iup[i]],1.1);
     if (topo[j][i]>topo[j][idown[i]]) 
      tot+=pow(topo[j][i]-topo[j][idown[i]],1.1);
     if (topo[j][i]>topo[jup[j]][i]) 
      tot+=pow(topo[j][i]-topo[jup[j]][i],1.1);
     if (topo[j][i]>topo[jdown[j]][i]) 
      tot+=pow(topo[j][i]-topo[jdown[j]][i],1.1);
     if (topo[j][i]>topo[jup[j]][iup[i]]) 
      tot+=pow((topo[j][i]-topo[jup[j]][iup[i]])*oneoversqrt2,1.1);
     if (topo[j][i]>topo[jdown[j]][iup[i]]) 
      tot+=pow((topo[j][i]-topo[jdown[j]][iup[i]])*oneoversqrt2,1.1);
     if (topo[j][i]>topo[jup[j]][idown[i]]) 
      tot+=pow((topo[j][i]-topo[jup[j]][idown[i]])*oneoversqrt2,1.1);
     if (topo[j][i]>topo[jdown[j]][idown[i]]) 
      tot+=pow((topo[j][i]-topo[jdown[j]][idown[i]])*oneoversqrt2,1.1);
     if (topo[j][i]>topo[j][iup[i]]) 
      flow1[j][i]=pow(topo[j][i]-topo[j][iup[i]],1.1)/tot; 
       else flow1[j][i]=0;
     if (topo[j][i]>topo[j][idown[i]]) 
      flow2[j][i]=pow(topo[j][i]-topo[j][idown[i]],1.1)/tot; 
       else flow2[j][i]=0;
     if (topo[j][i]>topo[jup[j]][i]) 
      flow3[j][i]=pow(topo[j][i]-topo[jup[j]][i],1.1)/tot; 
       else flow3[j][i]=0;
     if (topo[j][i]>topo[jdown[j]][i]) 
      flow4[j][i]=pow(topo[j][i]-topo[jdown[j]][i],1.1)/tot; 
       else flow4[j][i]=0;
     if (topo[j][i]>topo[jup[j]][iup[i]]) 
      flow5[j][i]=pow((topo[j][i]-topo[jup[j]][iup[i]])*oneoversqrt2,1.1)/tot;
       else flow5[j][i]=0;
     if (topo[j][i]>topo[jdown[j]][iup[i]]) 
      flow6[j][i]=pow((topo[j][i]-topo[jdown[j]][iup[i]])*oneoversqrt2,1.1)/tot;
       else flow6[j][i]=0;
     if (topo[j][i]>topo[jup[j]][idown[i]]) 
      flow7[j][i]=pow((topo[j][i]-topo[jup[j]][idown[i]])*oneoversqrt2,1.1)/tot;
       else flow7[j][i]=0;
     if (topo[j][i]>topo[jdown[j]][idown[i]]) 
      flow8[j][i]=pow((topo[j][i]-topo[jdown[j]][idown[i]])*oneoversqrt2,1.1)/tot;
       else flow8[j][i]=0;
     flow[j][iup[i]]+=flow[j][i]*flow1[j][i];
     flow[j][idown[i]]+=flow[j][i]*flow2[j][i];
     flow[jup[j]][i]+=flow[j][i]*flow3[j][i];
     flow[jdown[j]][i]+=flow[j][i]*flow4[j][i];
     flow[jup[j]][iup[i]]+=flow[j][i]*flow5[j][i];
     flow[jdown[j]][iup[i]]+=flow[j][i]*flow6[j][i];
     flow[jup[j]][idown[i]]+=flow[j][i]*flow7[j][i];
     flow[jdown[j]][idown[i]]+=flow[j][i]*flow8[j][i];
}

void calculatealongchannelslope(int i, int j)
{    double down;

     down=0;
     if (topo[j][iup[i]]-topo[j][i]<down) down=topo[j][iup[i]]-topo[j][i];
     if (topo[j][idown[i]]-topo[j][i]<down) down=topo[j][idown[i]]-topo[j][i];
     if (topo[jup[j]][i]-topo[j][i]<down) down=topo[jup[j]][i]-topo[j][i];
     if (topo[jdown[j]][i]-topo[j][i]<down) down=topo[jdown[j]][i]-topo[j][i];
     if ((topo[jup[j]][iup[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jup[j]][iup[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jup[j]][idown[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jup[j]][idown[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jdown[j]][iup[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jdown[j]][iup[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jdown[j]][idown[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jdown[j]][idown[i]]-topo[j][i])*oneoversqrt2;
     slope[j][i]=fabs(down)/dx;
}

void calculatealongchannelslope3(int i, int j)
{    double down;

     down=0;
     if (topo[j][iup3[i]]-topo[j][i]<down) down=topo[j][iup3[i]]-topo[j][i];
     if (topo[j][idown3[i]]-topo[j][i]<down) down=topo[j][idown3[i]]-topo[j][i];
     if (topo[jup3[j]][i]-topo[j][i]<down) down=topo[jup3[j]][i]-topo[j][i];
     if (topo[jdown3[j]][i]-topo[j][i]<down) down=topo[jdown3[j]][i]-topo[j][i];
     if ((topo[jup3[j]][iup3[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jup3[j]][iup3[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jup3[j]][idown3[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jup3[j]][idown3[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jdown3[j]][iup3[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jdown3[j]][iup3[i]]-topo[j][i])*oneoversqrt2;
     if ((topo[jdown3[j]][idown3[i]]-topo[j][i])*oneoversqrt2<down)
      down=(topo[jdown3[j]][idown3[i]]-topo[j][i])*oneoversqrt2;
     slope3[j][i]=fabs(down)/(3*dx);
}

void calculatemaxslope(int i, int j)
{
	maxslope[j][i]=slope[j][i];
	if (slope3[j][i]>maxslope[j][i]) maxslope[j][i]=slope3[j][i];
}

void check_command_args(int argc, char *argv[])
{

    if (argc != 2)
    {
        printf("Wrong number of command line arguments; expected 1, got %d; try again with: %s /path/to/work/dir\n", argc-1, argv[0]);
        exit(1);
    }

    DIR* dir = opendir(argv[1]);
    if (dir)
        /* Directory exists. */
        closedir(dir);
    else if (ENOENT == errno)
    {
        /* Directory does not exist. */
        printf("Work directory does not exist: %s\n", argv[1]);
        exit(1);
    }
    else
    {
        /* Some other error with opendir() */
        printf("Unkown error with opendir(%s)\n", argv[1]);
        exit(1);
    }

    return;
}

FILE *open_file(char fdir[], char fname[], const char *mode)
{
    FILE *fp;
    size_t len1 = strlen(fdir);
    size_t len2 = strlen(fname);
    char path_name[len1+len2+10];  // +10 just to be 100% sure we avoid buffer overflow

    // Concatenate directory and file name, then try to open it
    strcpy(path_name, fdir);
    strcat(path_name, fname);
    fp = fopen(path_name, mode);
    if (fp)
        return fp;
    else // Something went wrong, exit
    {
        printf("Error opening file: %s\n", path_name);
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    // Run this program's compiled executable from command line as: path/to/ProDF/executable/ProDF_..._.exe <work directory>
    // where <work directory> is the path to the work directory for both input and output txt files

    FILE *fr0,*fr1,*fp1,*fp2,*fp3;
    FILE *fr2;
    double *topovec,*startinfo;
    double nanval,c1,c2,g,rho,mu,tauy,xi;
    int i,j,k,t,*topovecind,exitstat;

    // Check command line argument
    check_command_args(argc, argv);

    exitstat=0;
    fp1 = open_file(argv[1], "/depthoutfinal.txt", "w");
    fp2 = open_file(argv[1], "/slopeout.txt", "w");
    fp3 = open_file(argv[1], "/exitstatus.txt", "w");
    
    startinfo=vector(1,11);
    fr0 = open_file(argv[1], "/startinfo.txt","r");
    for (i=1;i<=11;i++)
        fscanf(fr0,"%lf\n",&startinfo[i]);
    
    dx=startinfo[1];
    nx=startinfo[2];
    ny=startinfo[3];
    nanval=startinfo[4];
    c1=startinfo[5];
    c2=startinfo[6];
    g=startinfo[7];
    rho=startinfo[8];
    mu=startinfo[9];
    tauy=startinfo[10];
    xi=startinfo[11];
    
    setupgridneighbors();
    setupgridneighbors3();
    
    topo=matrix(1,ny,1,nx);
    itopo=matrix(1,ny,1,nx);
    toposave=matrix(1,ny,1,nx);
    topovec=vector(1,nx*ny);
    topovecind=ivector(1,nx*ny);
    volume=matrix(1,ny,1,nx);
    depth=matrix(1,ny,1,nx);
    slope=matrix(1,ny,1,nx);
    slope3=matrix(1,ny,1,nx);
    avgslope=matrix(1,ny,1,nx);
    maxslope=matrix(1,ny,1,nx);
    flow=matrix(1,ny,1,nx);
    flow1=matrix(1,ny,1,nx);
    flow2=matrix(1,ny,1,nx);
    flow3=matrix(1,ny,1,nx);
    flow4=matrix(1,ny,1,nx);
    flow5=matrix(1,ny,1,nx);
    flow6=matrix(1,ny,1,nx);
    flow7=matrix(1,ny,1,nx);
    flow8=matrix(1,ny,1,nx);
    
    time_t t0 = time(NULL);
    
    // Input Topography and Debris Flow Volume
    fr1 = open_file(argv[1], "/topoin.txt", "r");
    fr2 = open_file(argv[1], "/volumein.txt", "r");
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
        {
            fscanf(fr1,"%lf\n",&topo[j][i]);
            fscanf(fr2,"%lf\n",&volume[j][i]);
        }
    
    fclose(fr0);
    fclose(fr1);
    fclose(fr2);
    
    // Fill pits and flats to get hydrologically correct DEM
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
        {
            if (topo[j][i]!=nanval) fillinpitsandflats(i,j,nanval);
        }
    
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
        {
            itopo[j][i]=topo[j][i];
            calculatealongchannelslope(i,j);
            maxslope[j][i]=slope[j][i];
        }
    
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
        {
            calculatealongchannelslope3(i,j);
            calculatemaxslope(i,j);
        }
    
    // Set up topovec and do an initial quicksort
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
            topovec[(j-1)*nx+i]=topo[j][i];
    indexx(nx*ny, topovec, topovecind);

    for (k=1;k<=niterations;k++)
    {
        // Set the discharge that needs to be routed from each cell
        for (j=1;j<=ny;j++)
            for (i=1;i<=nx;i++)
            {
                toposave[j][i]=topo[j][i];
                flow[j][i]=c1*pow(volume[j][i],c2);    // discharge formula from Rickenmann (1999)
            }
        
        // Create index vectors and route flow
        insertsortx(nx*ny,topovec,topovecind); // Sort a vector in-place that contains elevation data ordered from highest to lowest
        t=nx*ny+1;                             // Set t equal to the total number of grid points plus one
        while (t>1)                            // Move through the vector we just created from highest to lowest elevation until we have visited every point
        {
            t--;
            i=(topovecind[t])%nx;              // Determine the row in the original grid where this data point lives
            if (i==0) i=nx;
            j=(topovecind[t])/nx+1;            // Determine the column in the original grid where this data point lives
            if (i==nx) j--;
            depth[j][i]=pow((flow[j][i]*3)/(2*dx*xi*(maxslope[j][i]+0.001)),0.4);    // Debris flow equation from Rickenmann (1999)
            if (rho*g*depth[j][i]*maxslope[j][i]>tauy) mfdflowroute(i,j);            // Stopping criteria based on yield strength
        }
        
        // Update routing surface
        for (j=1;j<=ny;j++)
            for (i=1;i<=nx;i++)
            {
                if ((itopo[j][i]+depth[j][i])>topo[j][i])
                {
                    topo[j][i]=toposave[j][i]+depth[j][i]/niterations;
                }
            }
        
        // Fill pits and flats to get hydrologically correct DEM
        for (j=1;j<=ny;j++)
            for (i=1;i<=nx;i++)
            {
                if (topo[j][i]!=nanval) fillinpitsandflats(i,j,nanval);
            }
        
        // Update the topo vector values for next in-place sort
        t=nx*ny+1;  
        while (t>1)
         {
             t--;
             i=(topovecind[t])%nx;
             if (i==0) i=nx;
             j=(topovecind[t])/nx+1;
             if (i==nx) j--;
             topovec[topovecind[t]] = topo[j][i];
         }
        
    }
    
    //Print Final Results
    for (j=1;j<=ny;j++)
        for (i=1;i<=nx;i++)
        {
            if (i==nx)
            {
                fprintf(fp1,"%lf\n",topo[j][i]-itopo[j][i]);         // Final Peak Flow Depth
                fprintf(fp2,"%lf\n",maxslope[j][i]);                 // Slope used in stopping criteria
            }
            if (i!=nx)
            {
                fprintf(fp1,"%lf\t",topo[j][i]-itopo[j][i]);         // Final Peak Flow Depth
                fprintf(fp2,"%lf\t",maxslope[j][i]);                 // Slope used in stopping criteria
            }
        }
    
    time_t t1 = time(NULL);
    printf ("Elapsed wall clock time: %ld\n", (long) (t1 - t0));
    
    fprintf(fp3,"%d\n",exitstat);
    
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
}
