/*
 * Copyright (c) 2014, OpenMote Technologies, S.L.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file is part of the Contiki operating system.
 *
 */
/*---------------------------------------------------------------------------*/
/**
 * \addtogroup openmote-cc2538
 * @{
 *
 * \defgroup openmote-examples OpenMote-CC2538 Example Projects
 * @{
 *
 * Example project demonstrating the OpenMote-CC2538 functionality
 *
 * @{
 *
 * \file
 * Example demonstrating the OpenMote-CC2538 platform
 * \author
 * Pere Tuset <peretuset@openmote.com>
 */
/*---------------------------------------------------------------------------*/

#include "contiki.h"
#include "cpu.h"
#include "sys/etimer.h"
#include "dev/leds.h"
#include "dev/uart.h"
#include "dev/serial-line.h"
#include "dev/button-sensor.h"
#include "dev/sys-ctrl.h"
#include "net/rime/broadcast.h"
#include "dev/sht21.h"
#include "dev/max44009.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*---------------------------------------------------------------------------*/
#define BROADCAST_CHANNEL   129
#define TRUE 1
#define FALSE 0
#define STATE 2 // n=2
#define OUTPUT 1 // m=1
#define INPUT 1 // u=1
#define THETA (STATE+STATE*INPUT+STATE*OUTPUT)// t = n + n * u + n * m,theta denotes all the parameters; order of Theta=A+B+K, not include C
/*---------------------------------------------------------------------------*/
PROCESS(openmote_pem, "OpenMote-CC2538 pem algorithem");
AUTOSTART_PROCESSES(&openmote_pem);
/*---------------------------------------------------------------------------*/
static void
broadcast_recv(struct broadcast_conn *c, const linkaddr_t *from)
{
  //this node does not receive
}
/*---------------------------------------------------------------------------*/
static const struct broadcast_callbacks broadcast_call = {broadcast_recv};
static struct broadcast_conn broadcast;
/*---------------------------------------------------------------------------*/
float det(float S[])//Determinant, S==one dimension array
{
	float result = 0;
	if (OUTPUT == 1)
	{
		result = S[0];
	}
	if (OUTPUT == 2)
	{
		float A[2][2] = { 0 };
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				A[i][j] = S[i*2 + j];
			}
		}
		result = A[0][0] * A[1][1] - A[0][1] * A[1][0];
	}
	if (OUTPUT > 2)
	{
		printf("array dimension out of bound\n");
	}
	return result;
}

//definition, parameters in PEM
static float Ahat[STATE*STATE] = { 0,1,0,0 };//A[4]
static float Ahat_old[STATE*STATE] = { 0,1,0,0 };
//static float Chat[OUTPUT*STATE] = { 1, 0 };
static float Chat_old[OUTPUT*STATE] = { 1, 0 };//Chat_old==Chat all the time, but denoted separately anyway!
static float Thehat[THETA] = { 0 };
static float Thehat_old[THETA] = { 0 };
static float Bhat[STATE*INPUT] = { 0 };
static float Khat[STATE*OUTPUT] = { 0 };
static float Khat_old[STATE*OUTPUT] = { 0 };
static float I[OUTPUT*OUTPUT] = { 1 };
float Xhatdot0[STATE*THETA] = { 0 };
static float Xhatdot[STATE*THETA] = { 0 };
static float Xhatdot_old[STATE*THETA] = { 0 };
static float P_old[THETA*THETA] = { 0 };// algorithm gain, P[k-1]
static float P_old2[THETA*THETA] = { 0 };//P[k-2]
static float E[OUTPUT] = { 0 };
static float E_old[OUTPUT] = { 0 };
static float Psi_old[THETA*OUTPUT] = { 0 };// regression，Psi[k-1]
static float Psi_old2[THETA*OUTPUT] = { 0 };//Psi[k-2]
static float Xhat[STATE] = { 0 };
static float Xhat_old[STATE] = { 0 };
static float Xhat_new[STATE] = { 0 };
static float Yhat_new[OUTPUT] = { 0 };
static float Yhat[OUTPUT] = { 0 };
static float Yhat_old[OUTPUT] = { 0 };
static float U[INPUT] = { 0 }; //come from environment=0
static float Y[OUTPUT] = { 0 };//come from sensor node!!
static float U_old[INPUT] = { 0 };
static float Y_old[OUTPUT] = { 0 };
static int16_t k = 0;
static float VN0 = 0;
static float VN = 0;

/*--------------------------------*/
int dgt(float a)//convert float part to string, 7 digits at most!
{
	
	int ipart=(int)a;
	float fpart = a- (float)ipart;
	int result= (int)(fpart * 100);//2 digit after point.	
	return result;
}
/*-----------------------------------------------------------------------*/
PROCESS_THREAD(openmote_pem, ev, data)
{
	static struct etimer et;
	static uint16_t  sht21_present;
	static int16_t Tem, Humi;

	PROCESS_EXITHANDLER(broadcast_close(&broadcast);)
	PROCESS_BEGIN();
	broadcast_open(&broadcast, BROADCAST_CHANNEL, &broadcast_call);
//initialization
	for (int i = 0; i < THETA*THETA; i++)
	{
		P_old2[i] = 0.0007;
	}
	for (int i = 0; i < THETA*OUTPUT; i++)
	{
		Psi_old2[i] = 0.73;
	}
	for(int i=0; i<STATE;i++)
	{
		Xhat_old[i] = 0.1;
	}
	for(int i=0; i<THETA;i++)
	{
		Thehat_old[i]=0.1;
	}
/* Initialize the SHT21 sensor */
	sht21_present = SENSORS_ACTIVATE(sht21);
	if(sht21_present == SHT21_ERROR) 
	{
		printf("SHT21 sensor is NOT present!\n");
		leds_on(LEDS_ORANGE);
	}
//assign theta-hat//Ahat
	for (size_t a = STATE * (STATE - 1); a < STATE*STATE; a++)//a=2,3
	{
		Ahat[a] = Thehat[a - STATE* (STATE - 1)];
		Ahat_old[a] = Thehat_old[a - STATE * (STATE - 1)];
	}
//Bhat=(n,u)
	for (int b = 0; b < STATE; b++)//b=0,1
	{
		for (int b1 = 0; b1 < INPUT; b1++)//b1=0
		{
			Bhat[b*INPUT + b1] = Thehat[STATE + b * INPUT + b1];
		}
	}
//Khat,Khat_old(n,m)
	for (int h = 0; h < STATE; h++)//h=0,1
	{
		for (int h1 = 0; h1 < OUTPUT; h1++)//h1=0
		{
			Khat[h*OUTPUT + h1] = Thehat[STATE + STATE * INPUT + h * OUTPUT + h1];
			Khat_old[h*OUTPUT + h1] = Thehat_old[STATE + STATE * INPUT + h * OUTPUT + h1];
		}
	}
	packetbuf_clear();

while(1) 
{
	etimer_set(&et, CLOCK_SECOND*5);
	PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&et));

	Tem = sht21.value(SHT21_READ_TEMP);
	char read[10];
	sprintf(read,"%u.%u",Tem/100, Tem%100);
	Y[0]=atof(read);
/*
	Humi = sht21.value(SHT21_READ_RHUM);
	char read1[10];
	sprintf(read1, "%u.%u", Humi/100, Humi%100);
	Y[0]=atof(read1);*/
	leds_toggle(LEDS_RED);
/*----------recursive prediction errors algorithm__PEM---------------------*/
	//E
	for (int i = 0; i < OUTPUT; i++)
	{		
		E[i] = Y[i] - Yhat[i];
	}
	for (int i0 = 0; i0 < STATE; i0++)//derivative of A, i0=0, 1
	{
		Xhatdot0[THETA*(STATE - 1) + i0] = Xhat_old[i0];
	}
	for (int i1 = 0; i1 < STATE; i1++)//derivative of B(n*u)
	{
		for (int i11 = 0; i11 < INPUT; i11++)
		{
			Xhatdot0[i1*THETA + STATE + i1 * INPUT + i11] = U_old[i11];
		}
	}
	for (int i2 = 0; i2 < STATE; i2++)//derivative of K(n*m)
	{
		for (int i21 = 0; i21 < OUTPUT; i21++)
		{
			Xhatdot0[i2*THETA + STATE + STATE * INPUT + i2 * OUTPUT + i21] = E_old[i21];
		}
	}
	// @1
	//trans(Psi_old2)
	float transPsi_old2[OUTPUT*THETA] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			transPsi_old2[THETA*j + i] = Psi_old2[i*OUTPUT + j];
		}
	}
	//Khat_old * trans(Psi_old2)=multiKP
	float multiKP[STATE*THETA] = { 0 };
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			for (int d = 0; d < OUTPUT; d++)
			{
				multiKP[i*THETA + j] += Khat_old[i*OUTPUT + d] * transPsi_old2[d*THETA + j];
			}
		}
	}
	//multi(Ahat_old, Xhatdot_old)=multiAXd
	float multiAXd[STATE*THETA] = { 0 };
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			for (int d = 0; d < STATE; d++)
			{
				multiAXd[i*THETA + j] += Ahat_old[i*STATE + d] * Xhatdot_old[d*THETA + j];
			}
		}
	}
	//Xhatdot=Xhatdot0+multiAXd-multiKP
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			Xhatdot[i*THETA + j] = Xhatdot0[i*THETA + j] + multiAXd[i*THETA + j] - multiKP[i*THETA + j];
		}
	}

	//@2,multi(Chat_old, Xhatdot)=multiCX
	float multiCX[OUTPUT*THETA] = { 0 };
	for (int i = 0; i < OUTPUT; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			for (int d = 0; d < STATE; d++)
			{
				multiCX[i*THETA + j] += Chat_old[i*STATE + d] * Xhatdot[d*THETA + j];
			}
		}
	}
	//Psi_old=trans(multiCX)
	for (int i = 0; i < OUTPUT; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			Psi_old[j*OUTPUT + i] = multiCX[i*THETA + j];
		}
	}

	//@3//trans(Psi_old)
	float transPsi_old[OUTPUT*THETA] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			transPsi_old[THETA*j + i] = Psi_old[i*OUTPUT + j];
		}
	}
	//trans(Psi_old)*P_old2=multiTPP
	float multiTPP[OUTPUT*THETA] = { 0 };
	for (int i = 0; i < OUTPUT; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			for (int d = 0; d < THETA; d++)
			{
				multiTPP[i*THETA + j] += transPsi_old[i*THETA + d] * P_old2[d*THETA + j];
			}
		}
	}
	//multiTPP*Psi_old=multiTPPP
	float multiTPPP[OUTPUT*OUTPUT] = { 0 };
	for (int i = 0; i < OUTPUT; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			for (int d = 0; d < THETA; d++)
			{
				multiTPPP[i*OUTPUT + j] += multiTPP[i*THETA + d] * Psi_old[d*OUTPUT + j];
			}
		}
	}
	//J=I+multiTPPP
	float J[OUTPUT*OUTPUT] = { 0 };
	for (int i = 0; i < OUTPUT; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			J[i*OUTPUT + j] = I[i*OUTPUT + j] + multiTPPP[i*OUTPUT + j];
		}
	}

	//@4//invJ
	float invJ[OUTPUT*OUTPUT] = { 0 };
	if (det(J) == 0)
	{
		printf("singularmatrix");
	}
	if (OUTPUT == 1)
	{
		invJ[0] = 1 / J[0];
	}
	if (OUTPUT == 2)
	{
		float result[2][2] = { 0 };
		float A[2][2] = { 0 };
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				A[i][j] = J[i*2 + j];
			}
		}
		float result0[2][2];
		result0[0][0] = A[1][1];
		result0[0][1] = -A[0][1];
		result0[1][0] = -A[1][0];
		result0[1][1] = A[0][0];
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				result[i][j] = result0[i][j] / det(J);
			}
		}
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				invJ[i*2 + j] = result[i][j];
			}
		}
	}
	if (OUTPUT > 2)
	{
		printf("inverse of matrix:array dimension out of bound\n");
	}
	//multi(P_old2, Psi_old)==multiP2P
	float multiP2P[THETA*OUTPUT] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			for (int d = 0; d < THETA; d++)
			{
				multiP2P[i*OUTPUT + j] += P_old2[i*THETA + d] * Psi_old[d*OUTPUT + j];
			}
		}
	}
	//multiP2P*invJ=multiPJ
	float multiPJ[THETA*OUTPUT] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			for (int d = 0; d < OUTPUT; d++)
			{
				multiPJ[i*OUTPUT + j] += multiP2P[i*OUTPUT + d] * invJ[d*OUTPUT + j];
			}
		}
	}
	//multiPJ*multiTPP=multiPall
	float multiPall[THETA*THETA] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			for (int d = 0; d < OUTPUT; d++)
			{
				multiPall[i*THETA + j] += multiPJ[i*OUTPUT + d] * multiTPP[d*THETA + j];
			}
		}
	}
	//P_old=P_old2-multiPall
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < THETA; j++)
		{
			P_old[i*THETA + j] = P_old2[i*THETA + j] - multiPall[i*THETA + j];
		}
	}
	//@5//multi(P_old, Psi_old)==multiPP
	float multiPP[THETA*OUTPUT] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			for (int d = 0; d < THETA; d++)
			{
				multiPP[i*OUTPUT + j] += P_old[i*THETA + d] * Psi_old[d*OUTPUT + j];
			}
		}
	}
	//multiPP*E==multiPPE
	float multiPPE[THETA] = { 0 };
	for (int i = 0; i < THETA; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			multiPPE[i] += multiPP[i*OUTPUT + j] * E[j];
		}
	}

	//@5Thehatprime=Thehat_old+multiPPE, ignore rank checking after k=10
	for (int i = 0; i < THETA; i++)
	{
		Thehat[i]= Thehat_old[i] + multiPPE[i];// 
	}
/*------------------------------------------------------------------*/
	
	//update A/B/Khat
	for (size_t a = STATE * (STATE - 1); a < STATE*STATE; a++)
	{
		Ahat[a] = Thehat[a - STATE* (STATE - 1)];
	}
	for (int b = 0; b < STATE; b++)
	{
		for (int b1 = 0; b1 < INPUT; b1++)
		{
			Bhat[b*INPUT + b1] = Thehat[STATE + b * INPUT + b1];
		}
	}
	for (int h = 0; h < STATE; h++)
	{
		for (int h1 = 0; h1 < OUTPUT; h1++)
		{
			Khat[h*OUTPUT + h1] = Thehat[STATE + STATE * INPUT + h * OUTPUT + h1];
		}
	}

	//@7 multiBU=Bhat*U 
	float multiBU[STATE] = { 0 };
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < INPUT; j++)
		{
			multiBU[i] += Bhat[i*INPUT + j] * U[j];
		}
	}
	//multi(Khat, E)==multiKE
	float multiKE[STATE] = { 0 };
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < OUTPUT; j++)
		{
			multiKE[i] += Khat[i*OUTPUT + j] * E[j];
		}
	}
	//multi(Ahat, Xhat)==multiAX
	float multiAX[STATE] = { 0 };
	for (int i = 0; i < STATE; i++)
	{
		for (int j = 0; j < STATE; j++)
		{
			multiAX[i] += Ahat[i*STATE + j] * Xhat[j];
		}
	}
	//Xhat_new =multiAX+multiBU+multiKE
	for (int i = 0; i < STATE; i++)
	{
		Xhat_new[i] = multiAX[i] + multiBU[i] + multiKE[i];
	}

	//@8 Yhat_new=Chat*Xhat_new
	Yhat_new[0]=Xhat_new[0];
	
	//update every parameter which is time involved
	for (int i = 0; i < STATE; i++)
	{
		Xhat_old[i] = Xhat[i];
		Xhat[i] = Xhat_new[i];
	}
	for (int i = 0; i < STATE*STATE; i++)
	{
		Ahat_old[i] = Ahat[i];
	}
	for (int i = 0; i < STATE*THETA; i++)
	{
		Xhatdot_old[i] = Xhatdot[i];
	}
	for (int i = 0; i < STATE*OUTPUT; i++)
	{
		Khat_old[i] = Khat[i];
	}
	for (int i = 0; i < THETA*THETA; i++)
	{
		P_old2[i] = P_old[i];
	}
	for (int i = 0; i < THETA*OUTPUT; i++)
	{
		Psi_old2[i] = Psi_old[i];
	}
	for (int i = 0; i < INPUT; i++)
	{
		U_old[i] = U[i];// useful if U != 0
	}
	for (int i = 0; i < THETA; i++)
	{
		Thehat_old[i] = Thehat[i];
	}

	//squared prediction errors
	float sqE = 0;
	for (int i = 0; i < OUTPUT; i++)
	{
		sqE += E[i] * E[i];
	}
	VN0 = VN0 + sqE;
	k++;
	VN = VN0 / k;

	for (int i = 0; i < OUTPUT; i++)
	{
		E_old[i] = E[i];
	}
	//update Y,Yhat
	for (int i = 0; i < OUTPUT; i++)
	{
		Y_old[i] = Y[i];
	}
	for (int i = 0; i < OUTPUT; i++)
	{
		Yhat_old[i] = Yhat[i];
		Yhat[i] = Yhat_new[i];
	}
/*------------- send out -------------*/
	size_t node[20]={0};// Y, Yhat, VN,k 

	for(int i=0;i<OUTPUT;i++)
	{
		node[i]=(int16_t)(Y[i]*100);
		node[OUTPUT+i]=(int16_t)(Yhat[i]*100);
	}
	node[OUTPUT+OUTPUT]=(size_t)(VN*100);
	node[OUTPUT+OUTPUT+1]=k;

        packetbuf_copyfrom(&node, sizeof(node));

	broadcast_send(&broadcast);
	leds_toggle(LEDS_YELLOW);	  
}

	PROCESS_END();
}


