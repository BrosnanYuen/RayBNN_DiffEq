#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_DiffEq;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
#[ignore]
fn test_ODE() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


	let n:u64 = 10;
	let steps:u64 = 10001;



    let A_dims = arrayfire::Dim4::new(&[10,10,1,1]);
    let mut A = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_A.csv",
    	
    );


    let D_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut D = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_D.csv",
    	
    );


    let tspan_dims = arrayfire::Dim4::new(&[1,10001,1,1]);
    let mut tspan = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_tspan.csv",
    	
    );

    tspan = arrayfire::transpose(&tspan, false);


    let x0_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut x0 = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_x0.csv",
    	
    );



    let xeval_dims = arrayfire::Dim4::new(&[10001,10,1,1]);
    let mut xeval = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_xeval.csv",
    	
    );




	let diffeq = |t: f64, x: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		D.clone() + arrayfire::matmul(x, &A, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE)
	};


	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0,
		tend: 100.0,
		tstep: 1E-5,
		rtol: 1E-15,
	    atol: 1.0,
		error_select: RayBNN_DiffEq::ODE::ODE45::error_type::TOTAL_ERROR
	};



	let starttime = std::time::Instant::now();



	let mut t = arrayfire::constant::<f64>(0.0,A_dims);
	let mut f = arrayfire::constant::<f64>(0.0,A_dims);
	let mut dfdt = arrayfire::constant::<f64>(0.0,A_dims);

	clusterdiffeq::diffeq::ode45_f64::linear_ode_solve(
		&x0
		,diffeq
		,&options
		,&mut t
		,&mut f
		,&mut dfdt
	);


	let xpred = clusterdiffeq::interpol::linear_f64::find(&t
		,&f
		,&dfdt
		,&tspan);


	let mut relerror = xpred - xeval.clone();
    relerror = relerror/xeval;
    relerror = arrayfire::abs(&relerror);
    let (maxerr,_) =  arrayfire::max_all(&relerror);


    assert!(maxerr  <= 2E-3);

}
