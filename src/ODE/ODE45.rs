use arrayfire;







const ODE45_C2_F64: f64 = 1.0/5.0;
const ODE45_A21_F64: f64 = 1.0/5.0;

const ODE45_C3_F64: f64 = 3.0/10.0;
const ODE45_A31_F64: f64 = 3.0/40.0;
const ODE45_A32_F64: f64 = 9.0/40.0;

const ODE45_C4_F64: f64 = 4.0/5.0;
const ODE45_A41_F64: f64 = 44.0/45.0;
const ODE45_A42_F64: f64 = -56.0/15.0;
const ODE45_A43_F64: f64 = 32.0/9.0;

const ODE45_C5_F64: f64 = 8.0/9.0;
const ODE45_A51_F64: f64 = 19372.0/6561.0;
const ODE45_A52_F64: f64 = -25360.0/2187.0;
const ODE45_A53_F64: f64 = 64448.0/6561.0;
const ODE45_A54_F64: f64 = -212.0/729.0;

const ODE45_C6_F64: f64 = 1.0;
const ODE45_A61_F64: f64 = 9017.0/3168.0;
const ODE45_A62_F64: f64 = -355.0/33.0;
const ODE45_A63_F64: f64 = 46732.0/5247.0;
const ODE45_A64_F64: f64 = 49.0/176.0;
const ODE45_A65_F64: f64 = -5103.0/18656.0;

const ODE45_C7_F64: f64 = 1.0;





const ODE45_B1_F64: f64 = 35.0/384.0;

const ODE45_B3_F64: f64 = 500.0/1113.0;
const ODE45_B4_F64: f64 = 125.0/192.0;
const ODE45_B5_F64: f64 = -2187.0/6784.0;
const ODE45_B6_F64: f64 = 11.0/84.0;




const ODE45_B1E_F64: f64 = 5179.0/57600.0;

const ODE45_B3E_F64: f64 = 7571.0/16695.0;
const ODE45_B4E_F64: f64 = 393.0/640.0;
const ODE45_B5E_F64: f64 = -92097.0/339200.0;
const ODE45_B6E_F64: f64 = 187.0/2100.0;
const ODE45_B7E_F64: f64 = 1.0/40.0;



pub enum error_type {
    TOTAL_ERROR,
    INDIVIDUAL_ERROR,
}



pub struct ODE45_Options<Z> {
    pub tstart: Z,
    pub tend: Z,
	pub tstep: Z,
	pub rtol: Z,
	pub atol: Z,
	pub error_select: error_type,
}





//First order system of linear ODE on single device
//initial: initial values of the diffeq in 1 row vector
//diffeq: function that produces derivative at t and x vector
//options: ODE tolerance settings
//Output: Output Spline vector (t_arr,f_arr,dfdt_arr)
pub fn solve<Z: arrayfire::FloatingPoint>(
	initial: &arrayfire::Array<Z>
	,diffeq: impl Fn(&arrayfire::Array<Z>, &arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,options: &ODE45_Options<Z>
	,out_t_arr: &mut arrayfire::Array<Z>
	,out_f_arr: &mut arrayfire::Array<Z>
	,out_dfdt_arr: &mut arrayfire::Array<Z>)
	{

	let var_num: u64 = initial.dims()[0];


	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	//ODE45 Constants
	let ODE45_C2 = arrayfire::constant::<f64>(ODE45_C2_F64,t_dims).cast::<Z>();

	let ODE45_A21 = arrayfire::constant::<f64>(ODE45_A21_F64,t_dims).cast::<Z>();

	let ODE45_C3 = arrayfire::constant::<f64>(ODE45_C3_F64,t_dims).cast::<Z>();

	let ODE45_A31 = arrayfire::constant::<f64>(ODE45_A31_F64,t_dims).cast::<Z>();

	let ODE45_A32 = arrayfire::constant::<f64>(ODE45_A32_F64,t_dims).cast::<Z>();

	let ODE45_C4 = arrayfire::constant::<f64>(ODE45_C4_F64,t_dims).cast::<Z>();

	let ODE45_A41 = arrayfire::constant::<f64>(ODE45_A41_F64,t_dims).cast::<Z>();

	let ODE45_A42 = arrayfire::constant::<f64>(ODE45_A42_F64,t_dims).cast::<Z>();

	let ODE45_A43 = arrayfire::constant::<f64>(ODE45_A43_F64,t_dims).cast::<Z>();

	let ODE45_C5 = arrayfire::constant::<f64>(ODE45_C5_F64,t_dims).cast::<Z>();

	let ODE45_A51 = arrayfire::constant::<f64>(ODE45_A51_F64,t_dims).cast::<Z>();

	let ODE45_A52 = arrayfire::constant::<f64>(ODE45_A52_F64,t_dims).cast::<Z>();

	let ODE45_A53 = arrayfire::constant::<f64>(ODE45_A53_F64,t_dims).cast::<Z>();

	let ODE45_A54 = arrayfire::constant::<f64>(ODE45_A54_F64,t_dims).cast::<Z>();

	let ODE45_C6 = arrayfire::constant::<f64>(ODE45_C6_F64,t_dims).cast::<Z>();

	let ODE45_A61 = arrayfire::constant::<f64>(ODE45_A61_F64,t_dims).cast::<Z>();

	let ODE45_A62 = arrayfire::constant::<f64>(ODE45_A62_F64,t_dims).cast::<Z>();

	let ODE45_A63 = arrayfire::constant::<f64>(ODE45_A63_F64,t_dims).cast::<Z>();

	let ODE45_A64 = arrayfire::constant::<f64>(ODE45_A64_F64,t_dims).cast::<Z>();

	let ODE45_A65 = arrayfire::constant::<f64>(ODE45_A65_F64,t_dims).cast::<Z>();

	let ODE45_C7 = arrayfire::constant::<f64>(ODE45_C7_F64,t_dims).cast::<Z>();

	let ODE45_B1 = arrayfire::constant::<f64>(ODE45_B1_F64,t_dims).cast::<Z>();

	let ODE45_B3 = arrayfire::constant::<f64>(ODE45_B3_F64,t_dims).cast::<Z>();

	let ODE45_B4 = arrayfire::constant::<f64>(ODE45_B4_F64,t_dims).cast::<Z>();

	let ODE45_B5 = arrayfire::constant::<f64>(ODE45_B5_F64,t_dims).cast::<Z>();

	let ODE45_B6 = arrayfire::constant::<f64>(ODE45_B6_F64,t_dims).cast::<Z>();

	let ODE45_B1E = arrayfire::constant::<f64>(ODE45_B1E_F64,t_dims).cast::<Z>();

	let ODE45_B3E = arrayfire::constant::<f64>(ODE45_B3E_F64,t_dims).cast::<Z>();

	let ODE45_B4E = arrayfire::constant::<f64>(ODE45_B4E_F64,t_dims).cast::<Z>();

	let ODE45_B5E = arrayfire::constant::<f64>(ODE45_B5E_F64,t_dims).cast::<Z>();

	let ODE45_B6E = arrayfire::constant::<f64>(ODE45_B6E_F64,t_dims).cast::<Z>();

	let ODE45_B7E = arrayfire::constant::<f64>(ODE45_B7E_F64,t_dims).cast::<Z>();




	//Create time start variables
	let mut t_cpu = vec![options.tstart.clone()];
	let mut t = arrayfire::Array::new(&t_cpu, t_dims).cast::<Z>();

	let mut t_F64 = arrayfire::real(&t).cast::<f64>();
	let mut t_cpu = vec!(f64::default();t_F64.elements());
	t_F64.host(&mut t_cpu);

	let mut tend_cpu = vec![options.tend.clone()];
	let mut tend = arrayfire::Array::new(&tend_cpu, t_dims).cast::<Z>();
	
	let mut tend_F64 = arrayfire::real(&tend).cast::<f64>();
	let mut tend_cpu = vec!(f64::default();tend_F64.elements());
	tend_F64.host(&mut tend_cpu);

	let mut tstep_cpu = vec![options.tstep.clone()];
	let mut tstep = arrayfire::Array::new(&tstep_cpu, t_dims).cast::<Z>();

	let mut tstep_F64 = arrayfire::real(&tstep).cast::<f64>();
	let mut tstep_cpu = vec!(f64::default();tstep_F64.elements());
	tstep_F64.host(&mut tstep_cpu);

	let mut rtol_cpu = vec![options.rtol.clone()];
	let mut rtol = arrayfire::Array::new(&rtol_cpu, t_dims).cast::<Z>();
	
	let mut rtol_F64 = arrayfire::real(&rtol).cast::<f64>();
	let mut rtol_cpu = vec!(f64::default();rtol_F64.elements());
	rtol_F64.host(&mut rtol_cpu);

	let mut atol_cpu = vec![options.atol.clone()];
	let mut atol = arrayfire::Array::new(&atol_cpu, t_dims).cast::<Z>();

	let mut atol_F64 = arrayfire::real(&atol).cast::<f64>();
	let mut atol_cpu = vec!(f64::default();atol_F64.elements());
	atol_F64.host(&mut atol_cpu);

	
	//let normctrl: bool = options.error_select.clone() ;
	let mut cur_point = initial.clone();


	//Calculate derivative k1
	let mut k1 = diffeq(&t, &cur_point);

	//Output array
	
	*out_t_arr = t.clone();
	*out_f_arr =  initial.clone();
	*out_dfdt_arr = k1.clone();



	let mut nerr: f64 = 1.0;
	let mut rerr: f64 = 1.0;
	let mut tol: f64 = 1.0;

	let cmp_dims = arrayfire::Dim4::new(&[var_num,2,1,1]);
	
	let mut cmparr = t.clone();


	match options.error_select {
		error_type::INDIVIDUAL_ERROR => {
			let mut atol_cpu2 = vec![options.atol.clone(); (2*var_num as usize) ];
			cmparr = arrayfire::Array::new(&atol_cpu2, cmp_dims).cast::<Z>();
		},
		error_type::TOTAL_ERROR => {
		}
	}


	let mut cmparr = cmparr.cast::<f64>();

	let mut tol_cpu: Vec<f64> = vec![1.0];
	let mut nerr_cpu: Vec<f64> = vec![1.0];




    let mut t2 = t.clone();
    let mut t3 = t.clone();
    let mut t4 = t.clone();
    let mut t5 = t.clone();
    let mut t6 = t.clone();
    let mut t7 = t.clone();



    let mut point2 = cur_point.clone();
    let mut point3 = cur_point.clone();
    let mut point4 = cur_point.clone();
    let mut point5 = cur_point.clone();
    let mut point6 = cur_point.clone();
    let mut point7 = cur_point.clone();




    let mut k2 = k1.clone();
    let mut k3 = k1.clone();
    let mut k4 = k1.clone();
    let mut k5 = k1.clone();
    let mut k6 = k1.clone();
    let mut k7 = k1.clone();


    let mut y0 = k1.clone();
    let mut y1 = k1.clone();
    let mut subtract = k1.clone();

	

    let mut abserror = arrayfire::constant::<f64>(0.0,t_dims);
    let mut absvec = arrayfire::constant::<f64>(0.0,t_dims);
    let mut minarr = arrayfire::constant::<f64>(0.0,t_dims);
    let mut result = arrayfire::constant::<f64>(0.0,t_dims);
    let mut tol_gpu = arrayfire::constant::<f64>(0.0,t_dims);
    let mut nerr_gpu = arrayfire::constant::<f64>(0.0,t_dims);



	let tend_cpu0 = tend_cpu[0].clone();

	let atol_cpu0 = atol_cpu[0].clone();

	let rtol_cpu0 = rtol_cpu[0].clone();

	while   t_cpu[0] <  tend_cpu0 {

		//Time arr 2
		t2 = t.clone() + (tstep.clone()*ODE45_C2.clone()) ;

		//Create point vector 2
		point2 = cur_point.clone() +  (tstep.clone()*ODE45_A21.clone()*(k1.clone()) );

		//Calculate derivative k2
		k2 = diffeq(&t2, &point2);






		//Time arr 3
		t3 = t.clone()  + (tstep.clone()*ODE45_C3.clone()) ;

		//Create point vector 3
		point3 = cur_point.clone() +  tstep.clone()*(  (ODE45_A31.clone()*(k1.clone()))   +  (ODE45_A32.clone()*(k2.clone()))    );

		//Calculate derivative k3
		k3 = diffeq(&t3,&point3) ;







		//Time arr 4
		t4 = t.clone() + (tstep.clone()*ODE45_C4.clone());

		//Create point vector 4
		point4 = cur_point.clone() +  tstep.clone()*(  (ODE45_A41.clone()*(k1.clone()))   +  (ODE45_A42.clone()*(k2.clone()))  +  (ODE45_A43.clone()*(k3.clone()))   );


		//Calculate derivative k4
		k4 = diffeq(&t4,&point4);






		//Time arr 5
		t5 = t.clone() + (tstep.clone()*ODE45_C5.clone()) ;

		//Create point vector 4
		point5 = cur_point.clone() +  tstep.clone()*(  (ODE45_A51.clone()*(k1.clone()))   +  (ODE45_A52.clone()*(k2.clone()))  +  (ODE45_A53.clone()*(k3.clone()))      +  (ODE45_A54.clone()*(k4.clone()))      );

		//Calculate derivative k5
		k5 = diffeq(&t5,&point5);








		//Time arr 5
		t6 = t.clone() + (tstep.clone()*ODE45_C6.clone());

		//Create point vector 4
		point6 = cur_point.clone() +  tstep.clone()*(  (ODE45_A61.clone()*(k1.clone()))   +  (ODE45_A62.clone()*(k2.clone()))  +  (ODE45_A63.clone()*(k3.clone()))      +  (ODE45_A64.clone()*(k4.clone()))   +  (ODE45_A65.clone()*(k5.clone()))     );

		//Calculate derivative k4
		k6 = diffeq(&t6,&point6);






		//Time arr 5
		t7 = t.clone() + (tstep.clone()*ODE45_C7.clone());

		//Create point vector 4
		//let point7 = cur_point.clone() +  tstep*(  (ODE45_A71*(k1.clone()))   +  (ODE45_A72*(k2.clone()))  +  (ODE45_A73*(k3.clone()))      +  (ODE45_A74*(k4.clone()))   +  (ODE45_A75*(k5.clone()))    +  (ODE45_A76*(k6.clone()))   );

		y0 = tstep.clone()*( (ODE45_B1.clone()*k1.clone())  +   (ODE45_B3.clone()*k3.clone()) +  (ODE45_B4.clone()*k4.clone()) +  (ODE45_B5.clone()*k5.clone()) +  (ODE45_B6.clone()*k6.clone())  );
		point7 = cur_point.clone() + y0.clone();

		//Calculate derivative k4
		k7 = diffeq(&t7,&point7);




		y1 = tstep*( (ODE45_B1E.clone()*k1.clone())   +  (ODE45_B3E.clone()*k3) +  (ODE45_B4E.clone()*k4) +  (ODE45_B5E.clone()*k5) +  (ODE45_B6E.clone()*k6) +  (ODE45_B7E.clone()*k7.clone()) );
		subtract = y1.clone() - y0.clone();




		match options.error_select {
			error_type::INDIVIDUAL_ERROR => {
				abserror = arrayfire::abs(&subtract).cast::<f64>();
				absvec = rtol_cpu0 * arrayfire::abs(&y0).cast::<f64>();

				arrayfire::set_col(&mut cmparr, &absvec,1);
				minarr = arrayfire::min(&cmparr,1);
				result = abserror.clone() - minarr.clone();


				let (_,_,idx) = arrayfire::imax_all(&result);

				tol_gpu = arrayfire::row(&minarr, idx as i64);
				nerr_gpu = arrayfire::row(&abserror, idx as i64);


				tol_gpu.host(&mut tol_cpu);
				tol = tol_cpu[0];


				nerr_gpu.host(&mut nerr_cpu);
				nerr = nerr_cpu[0];
			},
			error_type::TOTAL_ERROR => {
				nerr = arrayfire::norm::<Z>(&subtract,arrayfire::NormType::VECTOR_2,0.0,0.0  )   ;
				rerr = arrayfire::norm::<Z>(&y0,arrayfire::NormType::VECTOR_2,0.0,0.0  )  ;
				tol = atol_cpu0.min( rtol_cpu0*rerr );
			}
		}




		if  nerr < tol
		{
			//New point
			cur_point = point7.clone();

			//New time
			t = t7.clone();

			//New derivative
			k1 = k7.clone();

			

			//Save to array
			*out_t_arr = arrayfire::join::<Z>(1,out_t_arr,&t);

			*out_f_arr = arrayfire::join::<Z>(1,out_f_arr,&cur_point);

			*out_dfdt_arr = arrayfire::join::<Z>(1,out_dfdt_arr,&k1);

		}


		tstep_cpu[0] = 0.9*tstep_cpu[0]*( ( ( (tol/(nerr + 1E-30)).powf(0.2)).max(0.1)  ).min(2.0)  );



		//Update
		t_F64 = arrayfire::real(&t).cast::<f64>();
		t_F64.host(&mut t_cpu);


		tstep = arrayfire::Array::new(&tstep_cpu, t_dims).cast::<Z>();


	}


}
