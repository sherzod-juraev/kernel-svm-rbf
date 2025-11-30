from fastapi import APIRouter, status


modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='Fit training data',
    status_code=status.HTTP_200_OK,
    #response_model=
)
async def model_fit():
    pass


@modules_router.post(
    '/predict',
    summary='Predict',
    status_code=status.HTTP_200_OK,
    #response_model=
)
async def model_predict():
    pass