/* CLOMP Benchmark to measure OpenMP overheads with typical usage situations.
 * Also measures cache effects and memory bandwidth affects on performance.
 * Provides memory layout options that may affect performance.
 * Removes earlier variations that didn't provide expected insight.
 * Version 1.2 Drastically reduces serial overhead for high part count
 *             inputs, allowing reasonable messages with large thread counts.
 *             Also added -DWITH_MPI to measure OpenMP overheads when run
 *             with multiple MPI tasks per node, defaults to no MPI.
 *             Reduced digits reported after decimal point for most measurments.
 *             Last output line prints CORAL RFP data for easy importing.
 *             Changes by John Gyllenhaal at LLNL 12/23/13
 * Version 1.1 calculates error bounds (but not yet used) and dramatically
 *             reduces chance of underflows in math.  John Gyllenhaal 6/16/10
 * Version 1 (first public release) of the CLOMP benchmark.
 * Written by John Gyllenhaal and Greg Bronevetsky at LLNL 5/25/07
 * Based on version 0.2 (4/24/07) and version 0.1 (12/14/07) by John and Greg.
 ******************************************************************************
COPYRIGHT AND LICENSE

Copyright (c) 2007, The Regents of the University of California.
Produced at the Lawrence Livermore National Laboratory
Written by John Gyllenhaal (gyllen@llnl.gov)
and Greg Bronevetsky (bronevetsky1@llnl.gov)
UCRL-CODE-233406.
All rights reserved.

This file is part of CLOMP.
For details, see www.llnl.gov/asc/sequoia/benchmarks

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the disclaimer (as noted below) in
  the documentation and/or other materials provided with the distribution.

* Neither the name of the UC/LLNL nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY
OF CALIFORNIA, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ADDITIONAL BSD NOTICE

1. This notice is required to be provided under our contract with the
   U.S. Department of Energy (DOE). This work was produced at the
   University of California, Lawrence Livermore National Laboratory
   under Contract No. W-7405-ENG-48 with the DOE.

2. Neither the United States Government nor the University of California
   nor any of their employees, makes any warranty, express or implied,
   or assumes any liability or responsibility for the accuracy, completeness,
   or usefulness of any information, apparatus, product, or process disclosed,
   or represents that its use would not infringe privately-owned rights.

3. Also, reference herein to any specific commercial products, process,
   or services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or the University of California.
   The views and opinions of authors expressed herein do not necessarily
   state or reflect those of the United States Government or the University
   of California, and shall not be used for advertising or product
   endorsement purposes.
******************************************************************************/

extern "C"
{
#include <runtime/runtime.h>
}

#include "deref_scope.hpp"
#include "dataframe_vector.hpp"
#include "stats.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"

#include <stdio.h>
#include <stdlib.h>
//#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
//#ifdef WITH_MPI
// #include <mpi.h>
// #include <stdarg.h>
// int rank=0;
// int numtasks=0;
// #endif

using namespace far_memory;
using namespace std;

constexpr unsigned long CLOMP_numThreads = 1;    /* > 0 or -1 valid */
constexpr unsigned long CLOMP_allocThreads = 1;  /* > 0 or -1 valid */
constexpr unsigned long CLOMP_numParts = 16;     /* > 0 valid */
constexpr unsigned long CLOMP_zonesPerPart = 40; /* > 0 valid */
constexpr unsigned long CLOMP_zoneSize = 32;     /* > 0 valid, (sizeof(Zone) true min)*/
constexpr unsigned long CLOMP_flopScale = 1;     /* > 0 valid, 1 nominal */
constexpr unsigned long CLOMP_timeScale = 100;   /* > 0 valid, 100 nominal */
char *CLOMP_exe_name = NULL;                     /* Points to argv[0] */

/* Save actual argument for summary at end */
long CLOMP_inputAllocThreads = -2;

// DerefScope scope;

/* Simple Zone data structure */
class Zone
{
public:
    long zoneId;
    long partId;
    double value;
    SharedPtr<Zone> *nextZone = nullptr;
};

/* Part data structure */
class Part
{
public:
    unsigned long partId;
    unsigned long zoneCount;
    unsigned long update_count;
    // SharedPtr<Zone> *firstZone;
    // SharedPtr<Zone> *lastZone;
    SharedPtr<Zone> *firstZone = nullptr;
    SharedPtr<Zone> *lastZone = nullptr;

    double deposit_ratio;
    double residue;
    double expected_first_value; /* Used to check results */
    double expected_residue;     /* Used to check results */
};

/* Part array working on (now array of Part pointers)*/
// SharedPtr<Part> **partArray = NULL;

far_memory::Array<SharedPtr<Part>, CLOMP_numParts> *partArray;

constexpr static uint64_t kCacheSize = (128ULL << 20);
constexpr static uint64_t kFarMemSize = (4ULL << 30);
constexpr static uint32_t kNumGCThreads = 12;
constexpr static uint32_t kNumEntries = (8ULL << 20); // So the array size is larger than the local cache size.
constexpr static uint32_t kNumConnections = 300;

FarMemManager *far_mem_manager;

/* Used to avoid dividing by numParts */
double CLOMP_partRatio = 0.0;

/* Number of iterations to do approximately 0.5 to 2 second of work on a
 * fast new machine (deterministic calculated using heuristics).   */
long CLOMP_num_iterations = 0.0;

/* Used to check residue of checksum */
double CLOMP_max_residue = 0.0;

/* Used to calc_deposit's return value without having to check
 * every part's residue.  This allows more parts to be used and
 * thus higher thread counts.
 */
double CLOMP_residue_ratio_part0 = 0.0;

/* Used to determine if a calculation difference is a rounding error or
 * the result of the compiler, runtime, or test doing something bad
 * (i.e., races, bad barriers, etc.).
 * The loose error bound is calculated to be slightly smaller than
 * the smallest update of any zone.  This should catch races preventing
 * an update from happenning.
 */
double CLOMP_error_bound = 0.0;

/* The tight error bound is calculated to be the smallest difference between
 * two adjacent zone updates.   This should detect swapped updates (which
 * probably is not possible).   May be tighter than needed.
 */
double CLOMP_tightest_error_bound = 0.0;

void print_usage()
{
    fprintf(stderr,
            "Usage: clomp numThreads allocThreads numParts \\\n"
            "           zonesPerPart zoneSize flopScale timeScale\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "New in Version 1.2: Compile with -DWITH_MPI to generate clomp_mpi\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  numThreads: Number of OpenMP threads to use (-1 for system default)\n");
    fprintf(stderr, "  allocThreads: #threads when allocating data (-1 for numThreads)\n");
    fprintf(stderr, "  numParts: Number of independent pieces of work (loop iterations)\n");
    fprintf(stderr, "  zonesPerPart: Number of zones in the first part (3 flops/zone/part)\n");
    fprintf(stderr, "  zoneSize: Bytes in zone, only first ~32 used (512 nominal, >= 32 valid)\n");
    fprintf(stderr, "  flopScale: Scales flops/zone to increase memory reuse (1 nominal, >=1 Valid)\n");
    fprintf(stderr, "  timeScale: Scales target time per test (10-100 nominal, 1-10000 Valid)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Some interesting testcases (last number controls run time):\n");
    fprintf(stderr, "           Target input:    clomp 16 1 16 400 32 1 100\n");
    fprintf(stderr, "   Target/NUMA friendly:    clomp 16 -1 16 400 32 1 100\n");
    fprintf(stderr, "    Weak Scaling Target:    clomp N -1 N 400 32 1 100\n");
    fprintf(stderr, "      Weak Scaling Huge:    clomp N -1 N 6400 32 1 100\n");
    fprintf(stderr, "  Strong Scaling Target:    clomp -1 -1 1024 10 32 1 100\n");
    fprintf(stderr, "        Mem-bound input:    clomp N 1 N 640000 32 1 100\n");
    fprintf(stderr, "Mem-bound/NUMA friendly:    clomp N -1 N 640000 32 1 100\n");
    fprintf(stderr, "  MPI/OMP Hybrid Target:    (mpirun -np M) clomp_mpi 16 1 16 400 32 1 100\n");
}
/* Convert parm_val to a positive long value.  Punts if negative or not
 * completely translated
 */
long convert_to_positive_long(const char *parm_name, const char *parm_val)
{
    long val;
    char *endPtr;

    /* Sanity check */
    if ((parm_name == NULL) || (parm_val == NULL))
    {
        fprintf(stderr,
                "Error in convert_to_positive_long: Passed NULL pointers!\n");
        exit(1);
    }

    /* Convert string to long */
    val = strtol(parm_val, &endPtr, 0);

    /* Make sure everything in string was converted */
    if (endPtr[0] != 0)
    {
        fprintf(stderr, "Error converting '%s' parameter value '%s' to long at '%s'!\n",
                parm_name, parm_val, endPtr);
        exit(1);
    }

    /* Make sure final value > 0 except for numThreads and allocThreads,
     * which also takes -1
     */
    if ((strcmp(parm_name, "numThreads") == 0) ||
        (strcmp(parm_name, "allocThreads") == 0))
    {
        if ((val < 1) && (val != -1))
        {
            fprintf(stderr, "Invalid value %ld for parameter %s, must be > 0 or -1!\n",
                    val, parm_name);

            print_usage();
            exit(1);
        }
    }
    /* Make sure final value > 0 (except for numThreads checked above) */
    else if (val < 1)
    {
        fprintf(stderr, "Invalid value %ld for parameter %s, must be > 0\n",
                val, parm_name);

        print_usage();
        exit(1);
    }

    return (val);
}

void update_part(SharedPtr<Part> *part, double incoming_deposit)
{
    // SharedPtr<Zone> *zone;
    double deposit_ratio, remaining_deposit, deposit;
    long scale_count;

    {
        DerefScope scope;
        auto pPart = part->deref_mut(scope);

        // auto pZone = zone->deref_mut(scope);

        /* Update count of updates for this part (for error checking)
         * Just part 0's count will be zeroed regularly.   Others may wrap.
         */
        pPart->update_count++;

        /* Get the deposit_ratio from part*/
        deposit_ratio = pPart->deposit_ratio;

        /* Initially, the remaining_deposit is the incoming deposit */
        remaining_deposit = incoming_deposit;

        /* If have the most common case (original case) where CLOMP_flopScale = 1,
         * use a specialized loop to get best performance for this important case.
         * (Since the faster the loop, the more OpenMP overhead matters.)
         */
        if (CLOMP_flopScale == 1)
        {
            /* Run through each zone, depositing 'deposit_ratio' part of the
             * remaining_deposit in the zone and carrying the rest to the remaining
             * zones
             */

            for (auto zone = pPart->firstZone->deref_mut(scope); zone != NULL; zone = zone->nextZone->deref_mut(scope))
            {
                /* Calculate the deposit for this zone */
                deposit = remaining_deposit * deposit_ratio;

                /* Add deposit to the zone's value */
                zone->value += deposit;

                /* Remove deposit from the remaining_deposit */
                remaining_deposit -= deposit;
            }
        }

        /* Otherwise, if CLOMP_flopScale != 1, use inner loop version */
        else
        {
            /* Run through each zone, depositing 'deposit_ratio' part of the
             * remaining_deposit in the zone and carrying the rest to the remaining
             * zones
             */
            for (auto zone = pPart->firstZone->deref_mut(scope); zone != NULL; zone = zone->nextZone->deref_mut(scope))
            {
                /* Allow scaling of the flops per double loaded, so that you
                 * can get expensive iterations without blowing the cache.
                 */
                for (scale_count = 0; scale_count < CLOMP_flopScale; scale_count++)
                {
                    /* Calculate the deposit for this SharedPtr<Zone> */
                    deposit = remaining_deposit * deposit_ratio;

                    /* Add deposit to the zone's value */
                    zone->value += deposit;

                    /* Remove deposit from the remaining_deposit */
                    remaining_deposit -= deposit;
                }
            }
        }

        /* Put the left over deposit in the Part's residue field */
        pPart->residue = remaining_deposit;
    }
}

/* Resets parts to initial state and warms up cache */
void reinitialize_parts()
{
    cout << "reinitializing parts\n";
    unsigned long pidx;
    // SharedPtr<Zone> *zone;

    // DerefScope scope;

    /* Reset all the zone values to 0.0 and the part residue to 0.0 */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto &pointer_loc = partArray->at_mut(scope, pidx);
        auto part_ptr = &pointer_loc;
        auto partArray_index = part_ptr->deref_mut(scope);

        for (auto zone = partArray_index->firstZone->deref_mut(scope);
             zone != NULL;
             zone = zone->nextZone->deref_mut(scope))
        {
            /* Reset zone's value to 0 */
            zone->value = 0.0;
        }

        /* Reset residue */
        partArray_index->residue = 0.0;

        /* Reset update count */
        partArray_index->update_count = 0;
    }

    /* Scan through zones and add zero to each zone to warm up cache*/
    /* Also sets each zone update_count to 1, which sanity check wants */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArrayIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArrayIndexPtr, 0.0);
    }
}

/* Helper routine to print out timestamp when test starting */
void print_start_message(const char *desc)
{
    time_t starttime;
    char startdate[50]; /* Must be > 26 characters */

    /* Get time/data starting run*/
    std::time(&starttime);
    ctime_r(&starttime, startdate);

    /* Print out start message, startdate includes newline */
    printf("%13s  Started: %s", desc, startdate);

    /* Include number of threads if not "Serial Ref"or "calc_deposit"
     */
    if (strcmp(desc, "calc_deposit") != 0)
    {
        if (strcmp(desc, "Serial Ref") != 0)
        {
            /* Print out how many threads we are using */
            printf("%13s #Threads: %d\n", desc, 1);
        }
        else
        {
            /* Otherwise, print out N/A for Serial Ref */
            printf("%13s #Threads: N/A\n", desc);
        }
    }
}

/* Helper routine that puts time of day into passed ts structure */
void get_timestamp(struct timeval *ts)
{
    if (gettimeofday(ts, NULL) != 0)
    {
        fprintf(stderr, "Unable to get time of day, exiting\n");
        exit(1);
    }
}

/* Helper routine that prints a line of the benchmark pseudocode
 * prefixed with the description
 */
void print_pseudocode(const char *desc, const char *pseudocode)
{
    printf("%13s:| %s\n", desc, pseudocode);
}

/* Prints and returns run time in seconds (as double).
 * If base_seconds > 0.0, prints speedup numbers also
 */
double print_timestats(const char *desc, struct timeval *start_ts,
                       struct timeval *end_ts, double base_seconds,
                       double bestcase_seconds)
{
    double seconds;
    char used_desc[100] = "";

    /* Calculate run time */
    seconds = ((double)end_ts->tv_sec + ((double)end_ts->tv_usec * 1e-6)) -
              ((double)start_ts->tv_sec + ((double)start_ts->tv_usec * 1e-6));

    /* Print out overall runtime */
    printf("%13s  Runtime: %.3f (wallclock, in seconds%s)\n", desc, seconds,
           used_desc);

    /* Print runtime per subcycle in microseconds */
    printf("%13s  us/Loop: %.2f (wallclock, in microseconds)\n", desc,
           (seconds * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));

    /* If valid base time passed in and seconds valid, print speedups
     * (or slowdowns, if necessary)
     */
    if ((base_seconds > 0.0) && (seconds > 0.0))
    {
        if (base_seconds > seconds)
        {
            printf("%13s  Speedup: %.1f\n", desc, base_seconds / seconds);
        }
        else
        {
            printf("%13s  Speedup: %.1f (%.1fX slowdown)\n", desc,
                   base_seconds / seconds, seconds / base_seconds);
        }
    }

    if ((bestcase_seconds > 0.0) && (seconds > 0.0))
    {
        printf("%13s Efficacy: %.1f%% (of bestcase %.2f us/Loop)\n", desc,
               (bestcase_seconds / seconds) * 100.0,
               (bestcase_seconds * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));

        printf("%13s Overhead: %.2f (versus bestcase, in us/Loop)\n",
               desc, ((seconds - bestcase_seconds) * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));
    }

    printf("---------------------\n");

    return (seconds);
}

/* Check data for consistency and print out a one line data stat summary
 * that should help tell if data is all right  (total should be number
 * of iterations run).
 */
void print_data_stats(const char *desc)
{
    double value_sum, residue_sum, last_value, dtotal;
    long pidx;
    SharedPtr<Zone> *zone;
    int is_reference, error_count;

    /* Initialize value and residue sums to zero */
    value_sum = 0.0;
    residue_sum = 0.0;

    /* Use "Serial Ref" as the reference calculation and check all
     * values against that.  Since the same code is used to calculate
     * all these values, we should not have any issues with reordering
     * calculations causing small differences.
     */
    if (strcmp(desc, "Serial Ref") == 0)
        is_reference = 1;
    else
        is_reference = 0;

    /* Initialize count of check errors */
    error_count = 0;

    /* Scan through each part, check that values decrease monotonically
     * and sum up all the values.  Also check that the part residue and
     * the part's first zone's value are the expected value.
     */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        /* If have reference calculation, grab the first zone's value
         * and the part residue for comparison later
         */

        DerefScope scope;
        auto pArryIndexPointer = (&(partArray->at_mut(scope, pidx)))->deref_mut(scope);
        if (is_reference)
        {
            pArryIndexPointer->expected_first_value = pArryIndexPointer->firstZone->deref_mut(scope)->value;
            pArryIndexPointer->expected_residue = pArryIndexPointer->residue;
        }

        /* Otherwise, make sure this part matches the expected values
         * from the reference calculation.  Since using exactly the same
         * code to calculate these values (for all variations) and the
         * values are calculated in the same order (independent of number
         * of threads), I expect these values to be exactly the same.
         * If not true in the future, may have to put bounds on this.
         */
        else
        {
            /* Check that first zone's value is what is expected */
            if (pArryIndexPointer->expected_first_value != pArryIndexPointer->firstZone->deref_mut(scope)->value)
            {
                error_count++;
                fprintf(stderr,
                        "%s check failure: part %i first zone value (%g) != reference value (%g)!\n",
                        desc, (int)pidx, pArryIndexPointer->firstZone->deref_mut(scope)->value,
                        pArryIndexPointer->expected_first_value);
            }
            if (pArryIndexPointer->expected_residue != pArryIndexPointer->residue)
            {
                error_count++;
                fprintf(stderr,
                        "%s check failure: part %i residue (%g) != reference residue (%g)!\n",
                        desc, (int)pidx, pArryIndexPointer->residue,
                        pArryIndexPointer->expected_residue);
            }
        }

        /* Use first zone's value as initial last_value */
        last_value = pArryIndexPointer->firstZone->deref_mut(scope)->value;

        /* Scan through zones checking that values decrease monotonically */
        for (auto zone = pArryIndexPointer->firstZone->deref_mut(scope);
             zone != NULL;
             zone = zone->nextZone->deref_mut(scope))
        {
            if (zone->value > last_value)
            {
                fprintf(stderr,
                        "*** %s check failure (part %i zone %i): "
                        "previous (%g) < current (%g)!\n",
                        desc, (int)zone->partId,
                        (int)zone->zoneId, last_value, zone->value);
                error_count++;
            }

            /* Sum up values */
            value_sum += zone->value;

            /* This value now is last_value */
            last_value = zone->value;
        }

        /* Sum up part residue's */
        residue_sum += pArryIndexPointer->residue;
    }

    /* Calculate the total of value_sum + residue_sum.  This should
     * equal the number of subcycles run (due to construction of benchmark)
     */
    dtotal = value_sum + residue_sum;

    /* Sanity check, use reasonable bounds before alert */
    if (((dtotal + 0.00001) < ((double)CLOMP_num_iterations * 10.0)) ||
        ((dtotal - 0.00001) > ((double)CLOMP_num_iterations * 10.0)))
    {
        fprintf(stderr,
                "*** %s check failure:  Total (%-.15g) != Expected (%.15g)\n",
                desc, dtotal, ((double)CLOMP_num_iterations * 10.0));
        error_count++;
    }

    /* Sanity check, residue must be within reasonable bounds */
    if ((residue_sum < 0.0) || (residue_sum > (CLOMP_max_residue + 0.000001)))
    {
        fprintf(stderr,
                "*** %s check failure: Residue (%-.15g) outside bounds 0 - %.15g\n",
                desc, residue_sum, CLOMP_max_residue);
        error_count++;
    }

    /* Make sure part 0's update count is exactly one.  This detects
     * illegal optimization of calc_deposit().
     */
    DerefScope scope;
    auto pPartArryFirstIndexPtr = (&(partArray->at_mut(scope, pidx)))->deref_mut(scope);
    if (pPartArryFirstIndexPtr->update_count != 1)
    {
        fprintf(stderr, "Error in calc_deposit: Part updated %i times since last calc_deposit!\n",
                (int)pPartArryFirstIndexPtr->update_count);
        fprintf(stderr, "Benchmark designed to have calc_deposit called exactly once per update!\n");
        fprintf(stderr, "Critical error: Exiting...\n");
        exit(1);
    }

    if (error_count > 0)
    {
        fprintf(stderr,
                "ERROR: %i check failures detected in '%s' data. Exiting...\n",
                error_count, desc);
        exit(1);
    }

    /* Print out check text so results can be visually inspected */
    printf("%13s Checksum: Sum=%-8.8g Residue=%-8.8g Total=%-.9g\n",
           desc, value_sum, residue_sum, dtotal);
}

/* Calculates the amount to deposit in each part this subcycle.
 * Based on the residue of all parts from last subcycle.
 * Normally, this would be done with some sort of MPI exchange of
 * data but here we are just using info from local zones.
 *
 * Should be used by all subcycles to calculate amount to deposit.
 *
 * Cannot be moved past subcycle loops without getting wrong answer.
 */
double calc_deposit()
{
    double residue, deposit;
    // long pidx;

    /* Sanity check, make sure residues have be updated since last calculation
     * This code cannot be pulled out of loops or above loops!
     * Only check/update part 0 (other counts will just continue and may wrap)
     */
    DerefScope scope;
    auto pPartArryFirstIndexPtr = (&(partArray->at_mut(scope, 0)))->deref_mut(scope);
    if (pPartArryFirstIndexPtr->update_count != 1)
    {
        fprintf(stderr, "Error in calc_deposit: Part updated %i times since last call!\n",
                (int)pPartArryFirstIndexPtr->update_count);
        fprintf(stderr, "Benchmark designed to have calc_deposit called exactly once per update!\n");
        fprintf(stderr, "Critical error: Exiting...\n");
        exit(1);
    }

    /* Mark that we are using the updated info, so we can detect if this
     * code has been moved illegally.
     */
    pPartArryFirstIndexPtr->update_count = 0;

    /* Calculate residue from previous subcycle (normally this is done
     * with an MPI data exchange to other domains, but emulate here).
     */
    /* This method doesn't scale well past 64 parts with target input.
     * So using new method below to allow large part counts which
     * enables larger thread counts.
     *  -JCG 17Dec2013
     *
     * residue = 0.0;
     * for (pidx = 0; pidx < CLOMP_numParts; pidx++)
     * {
     *   residue += partArray[pidx]->residue;
     * }
     */

    /* Use direct calculation of deposit based on part 0's residue to
     * allow large part counts to be used.   Still want some data to
     * be used from calculation to prevent undesired optimizations.
     * Hopefully using part 0's data will be enough -JCG 17Dec2013
     */
    residue = pPartArryFirstIndexPtr->residue * CLOMP_residue_ratio_part0;

    /* Calculate deposit for this subcycle based on residue and part ratio */
    deposit = (1.0 + residue) * CLOMP_partRatio;

    /* Return the amount to deposit in each part this subcycle */
    return (deposit);
}

/* Do only the non-threadable calc_deposit() calls (without the update_part
 * loops) so that part of the benchmark can be timed separately.
 * Since we are not touching all the zone data, we run faster due to
 * having to cache much less data.  Unfortunately, this may make this
 * test run faster than when surrounded by the real code.
 */
void do_calc_deposit_only()
{
    cout << "do_calc_deposit_only\n";
    long iteration, subcycle;

    /* Do all the iterations */
    for (iteration = 0; iteration < CLOMP_num_iterations; iteration++)
    {
        /* 10 subcycles to every iteration, calc_deposit call in each one */
        for (subcycle = 0; subcycle < 10; subcycle++)
        {

            DerefScope scope;
            auto pPartArryFirstIndexPtr = (&(partArray->at_mut(scope, 0)))->deref_mut(scope);
            /* Fool calc_deposit sanity checks for this timing measurement */
            pPartArryFirstIndexPtr->update_count = 1;
            cout << "Updated count: " << pPartArryFirstIndexPtr->update_count << endl;

            /* Calc value, write into first zone's value, in order
             * to prevent compiler optimizing away
             */
            cout<<"pPartArryFirstIndexPtr->firstZone:"<<pPartArryFirstIndexPtr->firstZone<<endl;
            cout<<"pPartArryFirstIndexPtr->firstZone->deref_mut(scope):"<<pPartArryFirstIndexPtr->firstZone->deref_mut(scope)<<endl;
            pPartArryFirstIndexPtr->firstZone->deref_mut(scope)->value = calc_deposit();
        }
    }
}

/* Do module one's work serially (contains 1 subcycle) */
void serial_ref_module1()
{
    double deposit;
    long pidx;

    /* ---------------- SUBCYCLE 1 OF 1 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }
}

/* Do module two's work serially (contains 2 subcycles) */
void serial_ref_module2()
{
    double deposit;
    long pidx;

    /* ---------------- SUBCYCLE 1 OF 2 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 2 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }
}

/* Do module three's work serially (contains 3 subcycles) */
void serial_ref_module3()
{
    double deposit;
    long pidx;

    /* ---------------- SUBCYCLE 1 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 3 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }
}

/* Do module four's work serially (contains 4 subcycles) */
void serial_ref_module4()
{
    double deposit;

    /* ---------------- SUBCYCLE 1 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (auto pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (auto pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 3 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (auto pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }

    /* ---------------- SUBCYCLE 4 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's residue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (auto pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, pidx)));
        update_part(pPartArryIndexPtr, deposit);
    }
}

/* Do one cycle (10 subcycles) serially, no OpenMP */
void serial_ref_cycle()
{
    /* Emulate calls to 4 different packages, do 10 subcycles total */
    serial_ref_module1();
    serial_ref_module2();
    serial_ref_module3();
    serial_ref_module4();
}

/* Do all the cycles (10 subcycles/cycle) serially, no OpenMP */
void do_serial_ref_version()
{
    long iteration;

    /* Do the specified number of iterations */
    for (iteration = 0; iteration < CLOMP_num_iterations; iteration++)
        serial_ref_cycle();
}

/* Helper, add part passed in to the partArray at partId and initialize
 * it.  The separate routine is used to make it easy to allocate parts
 * with various strategies (such as each thread allocating it's parts).
 * The partArray has to be allocated by one thread but it is not
 * modified during the run.
 */
void addPart(SharedPtr<Part> *part, unsigned long partId)
{
    /* Sanity check, make sure partId valid */
    if ((partId < 0) || (partId >= CLOMP_numParts))
    {
        fprintf(stderr, "addPart error: partId (%i) out of bounds!\n", (int)partId);
        exit(1);
    }

    DerefScope scope;
    auto pPartArryIndexPtr = (&(partArray->at_mut(scope, partId)))->deref_mut(scope);

    /* Sanity check, make sure part not already added! */
    if (pPartArryIndexPtr != NULL)
    {
        fprintf(stderr, "addPart error: partId (%i) already initialized!\n",
                (int)partId);
        exit(1);
    }

    /* Put part pointer in array */
    partArray->at_mut(scope, partId) = *part;

    auto pPartObject = part->deref_mut(scope);

    /* Set partId */
    pPartObject->partId = partId;

    /* Set zone count for part (now fixed, used to be variable */
    pPartObject->zoneCount = CLOMP_zonesPerPart;

    /* Updated June 2010 by John Gyllenhaal to pick a deposit ratio
     * for this part that keeps the math from underflowing and
     * makes it possible to come up with a sane error bounds.
     * This math was picked experimentally to come up with
     * relatively large error bounds for the 'interesting testcase' inputs.
     *
     * This is part of an effort to separate rounding error due
     * to inlining update_part() (and optimizing different ways) from
     * incorrect results due to races or bad hardware.
     *
     * The older deposit_ratio only really worked well for 100 zones.
     */
    pPartObject->deposit_ratio = ((double)((1.5 * (double)CLOMP_numParts) + partId)) /
                                 ((double)(CLOMP_zonesPerPart * CLOMP_numParts));

    /* Initially no residue from previous passes */
    pPartObject->residue = 0.0;

    /* Initially, no zones attached to Part */
    pPartObject->firstZone = nullptr;
    pPartObject->lastZone = nullptr;

    /* Initially, don't know expected values (used for checking */
    pPartObject->expected_first_value = -1.0;
    pPartObject->expected_residue = -1.0;
}

/* Appends zone to the part identified by partId.   Done in separate routine
 * to facilitate the zones being allocated in various ways (such as by
 * different threads, randomly, etc.)
 */
void addZone(SharedPtr<Part> *part, SharedPtr<Zone> *zone)
{
    cout << "Stating part:" << part << endl;
    cout << "Stating zone:" << zone << endl;

    DerefScope scope;
    /* Sanity check, make sure not NULL */
    if (part->deref_mut(scope) == nullptr)
    {
        fprintf(stderr, "addZone error: part NULL!\n");
        exit(1);
    }

    /* Sanity check, make sure zone not NULL */
    if (zone->deref_mut(scope) == NULL)
    {
        fprintf(stderr, "addZone error: zone NULL!\n");
        exit(1);
    }

    /* Touch/initialize all of zone to force all memory to be really
     * allocated (CLOMP_zoneSize is often bigger than the portion of the
     * zone we use)
     */
    // memset(zone, 0xFF, CLOMP_zoneSize);

    cout << "Allocating new zone\n";
    auto *new_zone = new Zone;

    cout << "Setting new zone new zone\n";
    auto zonePtr = zone->deref_mut(scope);
    zonePtr = new_zone;

    /* If not existing zones, place at head of list */

    auto pPartObject = part->deref_mut(scope);
    cout << "pPartObject:" << pPartObject << endl;
    auto pZoneObject = zone->deref_mut(scope);
    cout << "pZoneObject:" << pZoneObject << endl;
    if (pPartObject->lastZone == NULL)
    {
        cout << "Entered in if block" << endl;
        cout << "pZoneObject->zoneId" << pZoneObject->zoneId << endl;
        /* Give first zone a zoneId of 1 */
        pZoneObject->zoneId = 1;

        /* First and last zone */
        pPartObject->firstZone = zone;
        pPartObject->lastZone = zone;
    }

    /* Otherwise, put after last zone */
    else
    {
        cout << "Entered in else block" << endl;
        /* Give this zone the last Zone's id + 1 */
        pZoneObject->zoneId = pPartObject->lastZone->deref_mut(scope)->zoneId + 1;

        pPartObject->lastZone->deref_mut(scope)->nextZone = zone;
        pPartObject->lastZone = zone;
    }

    /* Always placed at end */
    pZoneObject->nextZone = nullptr;

    /* Inialized the rest of the zone fields */
    pZoneObject->partId = pPartObject->partId;
    pZoneObject->value = 0.0;
}

/*
 * --------------------------------------------------------------------
 * Main driver
 * --------------------------------------------------------------------
 */

int argc;
std::string ip_addr_port;
void _main(void *arg)
{
    char **argv = static_cast<char **>(arg);
    char hostname[200];
    time_t starttime;
    char startdate[50]; /* Must be > 26 characters */
    long partId, zoneId;
    double totalZoneCount;
    double deposit, residue, percent_residue, part_deposit_bound;
    double deposit_diff_bound;
    double diterations;
    struct timeval calc_deposit_start_ts, calc_deposit_end_ts;
    double calc_deposit_seconds;
    struct timeval serial_ref_start_ts, serial_ref_end_ts;
    double serial_ref_seconds;
    int bidx, aidx;

    constexpr static uint64_t kCacheSize = (128ULL << 20);
    constexpr static uint64_t kFarMemSize = (4ULL << 30);
    constexpr static uint32_t kNumGCThreads = 12;
    // constexpr static uint32_t kNumEntries = (8ULL << 20); // So the array size is larger than the local cache size.
    constexpr static uint32_t kNumConnections = 300;

    auto raddr = helpers::str_to_netaddr(ip_addr_port);
    auto manager = std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
        kCacheSize, kNumGCThreads, new TCPDevice(raddr, kNumConnections, kFarMemSize)));

    // FarMemManager *far_mem_manager = manager.get();

    /* Get executable name by pointing to argv[0] */
    CLOMP_exe_name = argv[0];

    printf("CORAL Benchmark Version 1.2\n");

    /* Print usage if not 7 arguments */
    // if (argc != 8)
    // {
    //     print_usage();
    //     exit(1);
    // }

    /* Get hostname running on */
    if (gethostname(hostname, sizeof(hostname)) != 0)
        strcpy(hostname, "(Unknown host)");

    /* Get date starting benchmark on */
    std::time(&starttime);
    ctime_r(&starttime, startdate);

    /* Read in command line args (all must be positive ints) */
    // CLOMP_numThreads = convert_to_positive_long("numThreads", argv[1]);
    // CLOMP_allocThreads = convert_to_positive_long("numThreads", argv[2]);
    // CLOMP_numParts = convert_to_positive_long("numParts", argv[3]);
    // CLOMP_zonesPerPart = convert_to_positive_long("zonesPerPart", argv[4]);
    // CLOMP_zoneSize = convert_to_positive_long("zoneSize", argv[5]);
    // CLOMP_flopScale = convert_to_positive_long("flopScale", argv[6]);
    // CLOMP_timeScale = convert_to_positive_long("timeScale", argv[7]);

    /* Zone size cannot be less than sizeof(Zone), force to be valid */
    // if (CLOMP_zoneSize < sizeof(Zone))
    // {
    //     printf("***Forcing zoneSize (%ld specified) to minimum zone size %ld\n\n",
    //            CLOMP_zoneSize, (long)sizeof(Zone));
    //     //CLOMP_zoneSize = sizeof(Zone);
    // }

    /* Print out command line arguments as passed in */
    printf("       Invocation:");
    for (aidx = 0; aidx < argc; aidx++)
    {
        printf(" %s", argv[aidx]);
    }
    printf("\n");

    /* Print out command line arguments read in */
    printf("         Hostname: %s\n", hostname);
    printf("       Start time: %s", startdate); /* startdate has newline */
    printf("       Executable: %s\n", CLOMP_exe_name);
    if (CLOMP_numThreads == -1)
    {
        /* Set CLOMP_numThreads to system default if -1 */
        // CLOMP_numThreads = 1;

        /* Print out system default for threads */
        printf("      numThreads: %d (using system default)\n",
               (int)CLOMP_numThreads);
    }
    else
    {
        printf("      numThreads: %ld\n", CLOMP_numThreads);
    }

    /* Save actual argument for summary at end */
    CLOMP_inputAllocThreads = CLOMP_allocThreads;

    /* If -1, use numThreads for alloc threads */
    if (CLOMP_allocThreads == -1)
    {
        /* Use numThreads for alloc threads if -1 */
        // CLOMP_allocThreads = CLOMP_numThreads;

        /* Print out number of alloc threads to use */
        printf("    allocThreads: %ld (using numThreads)\n",
               CLOMP_allocThreads);
    }
    else
    {
        /* Print out number of alloc threads to use */
        printf("    allocThreads: %ld\n", CLOMP_allocThreads);
    }

    printf("        numParts: %ld\n", CLOMP_numParts);
    printf("    zonesPerPart: %ld\n", CLOMP_zonesPerPart);
    printf("       flopScale: %ld\n", CLOMP_flopScale);
    printf("       timeScale: %ld\n", CLOMP_timeScale);
    printf("        zoneSize: %ld\n", CLOMP_zoneSize);

    /* Allocate part pointer array */
    // PORT - initialize partArray with AIFM
    auto Temp_partArray = manager->allocate_array<SharedPtr<Part>, CLOMP_numParts>();
    partArray = &Temp_partArray;
    if (&partArray == nullptr)
    {
        fprintf(stderr, "Out of memory allocate part array\n");
        exit(1);
    }

    // PORT - initialize partArray to NULL

    for (partId = 0; partId < CLOMP_numParts; partId++)
    {
        DerefScope scope;
        auto arryIndexPtr = &(partArray->at_mut(scope, partId));
        arryIndexPtr = nullptr;
    }

    /* Calculate 1/numParts to prevent divides in algorithm */
    CLOMP_partRatio = 1.0 / ((double)CLOMP_numParts);

    /* Ininitialize parts (with no zones initially).
     * Do allocations in thread (allocThreads may be set to 1 for allocate)
     * to allow potentially better memory layout for threads
     */
    //#pragma omp parallel for private(partId) schedule(static)

    for (partId = 0; partId < CLOMP_numParts; partId++)
    {
        auto part = manager->allocate_shared_ptr<Part>();
        addPart(&part, partId);
        cout << "part added: " << partId << "\n";
    }
    cout << "part added Done\n";

    for (partId = 0; partId < CLOMP_numParts; partId++)
    {
        // far_memory::Array<SharedPtr<Zone>, CLOMP_zonesPerPart> *ZoneArray;

        uint64_t zoneId;

        for (zoneId = 0; zoneId < CLOMP_zonesPerPart; zoneId++)
        {
            auto zone = manager->allocate_shared_ptr<Zone>();
            DerefScope scope;
            addZone(&(partArray->at_mut(scope, partId)), &zone);
            cout << "Zone added: " << zoneId << "\n";
        }
    }

    /* Calculate the total number of zones */
    totalZoneCount = (double)CLOMP_numParts * (double)CLOMP_zonesPerPart;

    printf("   Zones per Part: %.0f\n", (double)CLOMP_zonesPerPart);
    printf("      Total Zones: %.0f\n", (double)totalZoneCount);
    printf("Memory (in bytes): %.0f\n", (double)(totalZoneCount * CLOMP_zoneSize) + (double)(sizeof(Part) * CLOMP_numParts));

    /* Calculate a number of iterations that experimentally appears to
     * give between 0.05 and 1.3 seconds of work on a 2GHz Opteron.
     * The 0.05 for small inputs that fit in cache, 1.3 for others.
     * The minimum iterations is 1, so once 1 million zones is hit,
     * the iterations cannot drop any more to compensate.
     * Also factor in flop_scaling now so setting flop_scaling to 100
     * doesn't make benchmark run 100X longer.
     */
    diterations = ceil((((double)1000000) * ((double)CLOMP_timeScale)) /
                       ((double)totalZoneCount * (double)CLOMP_flopScale));

    /* Sanity check for very small zone counts */
    if (diterations > 2000000000.0)
    {
        printf("*** Forcing iterations from (%g) to 2 billion\n",
               diterations);
        diterations = 2000000000.0;
    }

    /* Convert double iterations to int for use */
    CLOMP_num_iterations = (long)diterations;

    printf("Scaled Iterations: %i\n", (int)CLOMP_num_iterations);
    printf("  Total Subcycles: %.0f\n",
           (double)CLOMP_num_iterations * (double)10.0);

    /* Calculate serially the percent residue left after one pass */
    percent_residue = 0.0;

    /* If we deposit 1.0/numParts, sum of residues becomes percent residue */
    deposit = CLOMP_partRatio;

    /* In order to have sane rounding error bounds, determine the minimum
     * amount deposited in any zone.   Initialize to deposit as a starting
     * point for the min calculation below.
     */
    CLOMP_error_bound = deposit;
    CLOMP_tightest_error_bound = deposit;

    /* Scan through zones and add deposit to each zone */
    for (partId = 0; partId < CLOMP_numParts; partId++)
    {
        /* Do serial calculation of percent residue */

        DerefScope scope;
        auto pPartArryIndexPtr = (&(partArray->at_mut(scope, partId)));
        auto pPartArryIndexPtrAddress = pPartArryIndexPtr->deref_mut(scope);

        update_part(pPartArryIndexPtr, deposit);
        percent_residue += pPartArryIndexPtrAddress->residue;

        /* The 'next' deposit is smaller than any actual deposit made in
         * this part.  So it is a good lower bound on deposit size.
         */
        part_deposit_bound =
            pPartArryIndexPtrAddress->residue * pPartArryIndexPtrAddress->deposit_ratio;

        /* If we use the smallest bound on any part, we should have the
         * biggest error bound that will not give false negatives on
         * bad computation.
         */
        if (CLOMP_error_bound > part_deposit_bound)
        {
#if 0
	    printf ("After part %i bound from %g to %g (deposit %g)\n", (int)partId,
		    CLOMP_error_bound, part_deposit_bound, pPartArryIndexPtrAddress->deposit_ratio);
#endif
            CLOMP_error_bound = part_deposit_bound;
        }
#if 0
	else
	{
	    printf ("After part %i bound stayed %g because %g (deposit %g)\n", (int)partId,
		    CLOMP_error_bound, part_deposit_bound, pPartArryIndexPtrAddress->deposit_ratio);
	}
#endif

        deposit_diff_bound = part_deposit_bound * pPartArryIndexPtrAddress->deposit_ratio;
        if (CLOMP_tightest_error_bound > deposit_diff_bound)
        {
#if 0
	    printf ("After part %i tightest bound from %g to %g (deposit %g)\n", (int)partId,
		    CLOMP_tightest_error_bound, deposit_diff_bound, pPartArryIndexPtrAddress->deposit_ratio);
#endif
            CLOMP_tightest_error_bound = deposit_diff_bound;
        }
#if 0
	else
	{
	    printf ("After part %i tightest bound stayed %g because %g (deposit %g)\n", (int)partId,
		    CLOMP_tightest_error_bound, deposit_diff_bound, pPartArryIndexPtrAddress->deposit_ratio);
	}
#endif
    }
    printf("Iteration Residue: %.6f%%\n", percent_residue * 100.0);
    printf("  Max Error bound: %-8.8g\n", CLOMP_error_bound);
    printf("Tight Error bound: %-8.8g\n", CLOMP_tightest_error_bound);

#if 0
    printf ("DEBUG ENDING NOW\n");
    exit (1);
#endif

    /* Calculate the expected converged residue (for infinite iterations)
     * If Y is the percent residue after one pass, then the converged
     * expected converged residue = (added_each_cycle * Y)/ (1-Y)
     */
    CLOMP_max_residue = (1.0 * percent_residue) / (1 - percent_residue);
    printf("      Max Residue: %-8.8g\n", CLOMP_max_residue);

    /* In order to deal with huge thread counts, we need to make calc_deposit
     * not depend on the number of parts.   We do this by quickly caculating
     * an correct residue that still relies on at least part 0's data
     * (in an attempt to prevent undesired compiler optimizations).
     * Found part 0's residue is directly proportional to entire residue,
     * so calculate the ratio to get correct value. -JCG 17Dec2013
     */
    {
        DerefScope scope;
        auto pPartArryFirstIndexPtr = (&(partArray->at_mut(scope, 0)))->deref_mut(scope);
        CLOMP_residue_ratio_part0 = percent_residue / pPartArryFirstIndexPtr->residue;
    }

    /* Set the number of threads for the computation seciton to what user
     * specified. Because we are using alloc threads also, have to explicitly
     * set even if using system default
     */
    // omp_set_num_threads ((int)CLOMP_numThreads);

    /* Print initial line bar separator */
    printf("---------------------\n");

    /* --------- Start calc_deposit() overhead measurement --------- */

    /* DON'T REINITIALIZE PARTS BEFORE do_calc_deposit_only TEST!
     * We want there to be non-zero data in residues.
     */

    /* Calculate just the overhead of the non-threadable calc_deposit() calls.
     * May get good cache effects here, so may be low estimate.
     */
    print_pseudocode("calc_deposit", "------ Start calc_deposit Pseudocode ------");
    print_pseudocode("calc_deposit", "/* Measure *only* non-threadable calc_deposit() overhead.*/");
    print_pseudocode("calc_deposit", "/* Expect this overhead to be negligible.*/");
    print_pseudocode("calc_deposit", "deposit = calc_deposit ();");
    print_pseudocode("calc_deposit", "------- End calc_deposit Pseudocode -------");
    print_start_message("calc_deposit");

    get_timestamp(&calc_deposit_start_ts);
    do_calc_deposit_only();
    get_timestamp(&calc_deposit_end_ts);
    calc_deposit_seconds = print_timestats("calc_deposit",
                                           &calc_deposit_start_ts,
                                           &calc_deposit_end_ts, -1.0, -1.0);

    /* --------- Start Serial Ref benchmark measurement --------- */

    /* Do one cycle outside timer loop to warm up code and (for OpenMP cases)
     * allow the OpenMP system to initialize (which can be expensive, skewing
     * the measurments for small runtimes */
    reinitialize_parts();
    serial_ref_cycle();

    /* Reinitialize parts and warm up cache by doing dummy update */
    reinitialize_parts();

    /* Do the serial version of calculation and measure time*/
    print_pseudocode("Serial Ref", "------ Start Serial Ref Pseudocode ------");
    print_pseudocode("Serial Ref", "/* Measure serial reference performance */");
    print_pseudocode("Serial Ref", "deposit = calc_deposit ();");
    print_pseudocode("Serial Ref", "for (pidx = 0; pidx < numParts; pidx++)");
    print_pseudocode("Serial Ref", "  update_part (partArray[pidx], deposit);");
    print_pseudocode("Serial Ref", "------- End Serial Ref Pseudocode -------");
    print_start_message("Serial Ref");
    //#ifdef WITH_MPI
    /* Ensure all MPI tasks run OpenMP at the same time */
    // MPI_Barrier (MPI_COMM_WORLD);
    //#endif
    get_timestamp(&serial_ref_start_ts);
    do_serial_ref_version();
    get_timestamp(&serial_ref_end_ts);

    /* Check data for consistency and print out data stats*/
    print_data_stats("Serial Ref");

    /* Print out serial time stats and capture time,
     * -1 prevents speedup numbers from printing out
     */
    serial_ref_seconds = print_timestats("Serial Ref", &serial_ref_start_ts,
                                         &serial_ref_end_ts, -1.0, -1.0);

    printf("----------- Comma-delimited summary ----------\n");
    printf("%s %ld %ld %ld %ld %ld %ld %ld, calc_deposit, OMP Barrier, Serial Ref, Bestcase OMP, Static OMP, Dynamic OMP, Manual OMP\n",
           CLOMP_exe_name,
           CLOMP_numThreads,
           CLOMP_inputAllocThreads,
           CLOMP_numParts,
           CLOMP_zonesPerPart,
           CLOMP_zoneSize,
           CLOMP_flopScale,
           CLOMP_timeScale);

    printf("Runtime, %7.3f, %7.3f\n",
           calc_deposit_seconds,
           // omp_barrier_seconds,
           serial_ref_seconds);
    // bestcase_omp_seconds,
    // static_omp_seconds,
    // dynamic_omp_seconds,
    // manual_omp_seconds);

#undef us_loop
#define us_loop(sec) (((sec * 1000000.0) / ((double)CLOMP_num_iterations * 10.0)))
    printf("us/Loop, %7.2f, %7.2f\n",
           us_loop(calc_deposit_seconds),
           // us_loop(omp_barrier_seconds),
           us_loop(serial_ref_seconds));
    // us_loop(bestcase_omp_seconds),
    // us_loop(static_omp_seconds),
    // us_loop(dynamic_omp_seconds),
    // us_loop(manual_omp_seconds));

#undef speedup
#define speedup(sec) ((serial_ref_seconds / sec))
    printf("Speedup,     N/A,     N/A, %7.1f\n",
           speedup(serial_ref_seconds));

    {
        char mpi_marker[100] = "";
        printf("CORAL RFP, %s%ld %ld %ld %ld %ld %ld %ld, %.2f\n",
               mpi_marker,
               CLOMP_numThreads,
               CLOMP_inputAllocThreads,
               CLOMP_numParts,
               CLOMP_zonesPerPart,
               CLOMP_zoneSize,
               CLOMP_flopScale,
               CLOMP_timeScale,
               us_loop(serial_ref_seconds));
    }
}

int main(int _argc, char *argv[])
{
    int ret;

    if (_argc < 3)
    {
        std::cerr << "usage: [cfg_file] [ip_addr:port]" << std::endl;
        return -EINVAL;
    }

    char conf_path[strlen(argv[1]) + 1];
    strcpy(conf_path, argv[1]);
    ip_addr_port.assign(argv[2]);
    for (int i = 3; i < _argc; i++)
    {
        argv[i - 1] = argv[i];
    }
    argc = _argc - 2;

    // auto manager = std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
    //       kCacheSize, kNumGCThreads,  new TCPDevice(raddr, kNumConnections, kFarMemSize)));

    // far_mem_manager = manager.get();

    ret = runtime_init(conf_path, _main, argv);
    if (ret)
    {
        std::cerr << "failed to start runtime" << std::endl;
        return ret;
    }

    return 0;
}