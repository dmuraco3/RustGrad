#[macro_export]
macro_rules! tensor {
    (
        //5
        $([$([$([$([$($x:expr),* $(,)*]), + $(,)*]), + $(,)*]), + $(,)*]) , + $(,)*
    ) => {{
        {
            let t = vec![$(vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*],)*],)*];
            let dims = [
                t[0][0][0][0].len(),
                t[0][0][0].len(),
                t[0][0].len(),
                t[0].len(),
                5
            ]
        }
    }};
    (
        //4
        $([$([$([$($x:expr),* $(,)*]), + $(,)*]), + $(,)*]), + $(,)*
    ) => {{
        {
            let t = vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*],)*];
            let dims = [
                t[0][0][0].len(),
                t[0][0].len(),
                t[0].len(),
                4
            ]
        }
    }};
    (
        //3
        $d_type: ty,
        [$([$([$($x:expr),* $(,)*]), + $(,)*]), + $(,)*]
    ) => {{
        {
            let t = [$([$([$($x,)*],)*],)*];

            let dims = [
                t[0][0].len(),
                t[0].len(),
                3
            ];

            let dim_prod = dims.iter().fold(1, |sum, val| sum * val);


            let p = t.iter().flat_map(|x| x.iter()).flat_map(|x| x.iter()).cloned().collect::<Vec<$d_type>>();



            "bruh"

        }
    }};
    (
        //2
        $dtype: ty,
        [$([$($x:expr),* $(,)*]),+ $(,)*]
    ) => {{
        {
            let t = vec![$([$($x,)*],)*];

            let dims = [
                t[0].len(),
                2
            ];
            
            let dim_prod = dims.iter().fold(1, |sum, val| sum * val);



        }
    }};
    (
        //1
        $dtype: ty,
        [$($x:expr),* $(,)*]
    ) => {{
        {
            let t = vec![$($x,)*];
            let dims = [t.len()];
        }

    }};
    
}