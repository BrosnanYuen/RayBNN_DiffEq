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
    let mut A = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/ODE_A.csv",
    	A_dims
    );


    let D_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut D = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/ODE_D.csv",
    	D_dims
    );


    let tspan_dims = arrayfire::Dim4::new(&[1,10001,1,1]);
    let mut tspan = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/ODE_tspan.csv",
    	tspan_dims
    );

    tspan = arrayfire::transpose(&tspan, false);


    let x0_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut x0 = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/ODE_x0.csv",
    	x0_dims
    );



    let xeval_dims = arrayfire::Dim4::new(&[10001,10,1,1]);
    let mut xeval = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/ODE_xeval.csv",
    	xeval_dims
    );




	let diffeq = |t: f64, x: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		D.clone() + arrayfire::matmul(x, &A, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE)
	};


	let options: clusterdiffeq::diffeq::ode45_f64::ode45_f64_set = clusterdiffeq::diffeq::ode45_f64::ode45_f64_set {
		tstart: 0.0,
		tend: 100.0,
		tstep: 1E-5,
		rtol: 1E-15,
	    atol: 1.0,
		normctrl: true};


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
	,&mut dfdt);


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
